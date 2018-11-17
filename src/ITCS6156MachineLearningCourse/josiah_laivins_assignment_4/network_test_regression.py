import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from util import *

from LeeNeuralNet import NeuralNet

warnings.filterwarnings("ignore", category=DataConversionWarning)

""" Load data """
data_raw = pd.read_csv('./data/housing_prices/train.csv', nrows=None)
# Drop rows where (specific) columns are nan, and the Id column, and set the rest of nan to 0
data = data_raw.dropna(subset=list(set(data_raw.columns) - {'Alley', 'PoolQC', 'Fence', 'MiscFeature'}), axis=0)
data = data.drop(axis=1, columns='Id')
data = data.fillna('0')

""" Encode Dataframe """
encoder_d = defaultdict(LabelEncoder)

Y = data['SalePrice']
X = data.drop(axis=1, columns='SalePrice')  # type: pd.DataFrame
X = X.apply(lambda x: encoder_d[x.name].fit_transform(x))
X_value = np.array(X.values)
Y_value = np.array(Y.values).reshape(-1, 1)

""" Normalize Data """
standard_x_d = MinMaxScaler()
X_norm = standard_x_d.fit_transform(X.values)
standard_y = MinMaxScaler()
Y_norm = standard_y.fit_transform(Y.values.reshape(-1, 1))

print(f'Data loaded. Training X shape: {X_value.shape} Y shape: {Y_value.shape}')


""" Set up K-Fold """
k_folds = 2

# Logs for K and the params to test
train_params = [{'struct': [X_value.shape[1], 15, 1]}]
rmse_test_per_iter = []
rmse_train_per_iter = []
cost_log_per_iter = []
best_param_per_iter = []

""" Add unique ids for each param to be identified by """
for i in range(len(train_params)):
    train_params[i]['id'] = i

X_train, X_test, y_train, y_test = train_test_split(X_norm, Y_norm, test_size=0.20)
from_k = 0
# For each K
for k in tqdm(range(k_folds), bar_format='r_bar'):
    X_train_per_k = X_train[from_k:from_k+int(X_train.shape[0] / k_folds)]
    y_train_per_k = y_train[from_k:from_k+int(y_train.shape[0] / k_folds)]

    from_k += from_k+int(X_train.shape[0] / k_folds)

    # Test each param
    for param in train_params:
        """ Build Neural Net """
        nn = NeuralNet(param['struct'])
        """ Train Neural Network """
        nn.train(X_train_per_k, y_train_per_k, ftracep=True, wtracep=True)

        # Set test rmse local or res
        rmse_train_per_iter.append(nn.ftrace)
        best_param_per_iter.append(param)

""" Show the parameters and their performance """
plot_k_folds(k_folds, best_param_per_iter, extra_params=(rmse_train_per_iter,),
             labels=['K', 'RMSE for Train', 'Params'])
plt.show()

""" Keep the top 5 best """
import copy

top = 2

best_indices = np.array([_[-1] for _ in rmse_train_per_iter]).argsort()[:top]
copy_best_param_per_iter = copy.deepcopy([best_param_per_iter[i] for i in best_indices])
_, counts = np.unique([p['id'] for p in copy_best_param_per_iter], return_counts=True)

best_iter = best_indices[counts.argmax()]

# Copy the params because I dont want to accidentally reset them...
copy_rmse_train = rmse_train_per_iter[best_iter]

""" Plot the test and train RMSE """
plt.figure(figsize=(10,10))
plt.subplot(1,1,1)
plt.plot(copy_rmse_train)
plt.title(f'Best Param is: {get_formatted_params(best_param_per_iter[best_iter], include_values=True)} \n\n with final'+
          f' test RMSE of {copy_rmse_train[-1]}')
plt.xlabel('Epochs')
plt.ylabel('RMSE of best for Test and training')

plt.show()

nn = NeuralNet(best_param_per_iter[best_iter]['struct'])
""" Train Neural Network """
nn.train(X_train, y_train, ftracep=True, wtracep=True)

prediction_y = nn.use(X_test)

plt.subplot(3, 1, 2)
plt.plot(range(len(X_test)), y_test, 'o-', range(len(X_test)), prediction_y, 'o-')
plt.xlim(0, int(len(X_test)/2))
plt.legend(('Testing', 'Model'), loc='upper left')
plt.xlabel('$x$')
plt.ylabel('Actual and Predicted $f(x)$')


plt.show()