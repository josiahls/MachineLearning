import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm_notebook as tqdm

from NeuralNetworkLogisticRegression import NeuralNetLogReg
from TensorFlowDeepConvolutionalNetwork import TensorFlowDeepConvolutionalNetwork
from TensorFlowDeepForwardFeedNetwork import TensorFlowDeepForwardFeedNetwork
from util import *

# %matplotlib inline
from util import plot_confusion_matrix

""" Dictionary of the classification values and their string equivalent """
classification_name = {0: 'T-shirt/top',
                       1: 'Trouser',
                       2: 'Pullover',
                       3: 'Dress',
                       4: 'Coat',
                       5: 'Sandal',
                       6: 'Shirt',
                       7: 'Sneaker',
                       8: 'Bag',
                       9: 'Ankle boot'}

""" The train and test sets are already split """
train = pd.read_csv('./data/fashionmnist/fashion-mnist_train.csv', nrows=600)
test = pd.read_csv('./data/fashionmnist/fashion-mnist_test.csv', nrows=600)

""" These are images that are 28x28, the column 0 is the label """
pic_train = np.array(train.iloc[0, 1:]).reshape((28, 28))
pic_test = np.array(test.iloc[0, 1:]).reshape((28, 28))

""" Example of the data that will be used """
fig = plt.figure(figsize=(14, 14))
fig.add_subplot(1, 2, 1)
plt.title(f'{train.loc[0][0]} is a {classification_name[train.loc[0][0]]}')
plt.imshow(pic_train)
fig.add_subplot(1, 2, 2)
plt.title(f'{test.loc[0][0]} is a {classification_name[test.loc[0][0]]}')
plt.imshow(pic_test)
plt.show()

""" Convert Pandas Frames into X and Y """
combined_X = pd.concat((train, test)).reset_index(drop=True)

X = np.array(combined_X.iloc[:, 1:])
Y = np.array(combined_X.iloc[:, 0]).reshape(-1, 1)

""" Normalize and Standardize X and Y """
from Standardizer import Standardizer

standardizer_x = Standardizer(X, is_image=True)
stand_X = standardizer_x.standardize(X)
encoder = OneHotEncoder()
encoder.fit(Y)
stand_Y = encoder.transform(Y).toarray()


""" Set up K-Fold """
k_folds = 2

# Logs for K and the params to test
train_params = [
    # {'struct':[X[0].shape[0], 16, stand_Y.shape[1]]},
    # {'struct': [X[0].shape[0], 32, stand_Y.shape[1]]},
    {'struct': [X[0].shape[0], 32, 16, stand_Y.shape[1]]}
]

rmse_train_per_iter = []
train_accuracy = []
best_param_per_iter = []

""" Add unique ids for each param to be identified by """
for i in range(len(train_params)):
    train_params[i]['id'] = i

X_train, X_test, y_train, y_test = train_test_split(stand_X, stand_Y, test_size=0.20)
from_k = 0
# For each K
for k in tqdm(range(k_folds), bar_format='r_bar'):
    debug(f'Starting fold: {k}')
    X_train_per_k = X_train[from_k:from_k+int(X_train.shape[0] / k_folds)]
    y_train_per_k = y_train[from_k:from_k+int(y_train.shape[0] / k_folds)]

    from_k += from_k+int(X_train.shape[0] / k_folds)
    # Test each param
    for param in train_params:
        debug(f'Testing param: {param}')
        """ Build Neural Net """
        nn = TensorFlowDeepConvolutionalNetwork(param['struct'])

        """ Train Neural Network """
        nn.train(X_train_per_k, y_train_per_k, param)

        # Set test rmse local or res
        rmse_train_per_iter.append(nn.loss)
        train_accuracy.append(nn.accuracy)
        best_param_per_iter.append(param)

import copy
from util import *

""" Show the parameters and their performance """
plot_k_folds(k_folds, best_param_per_iter, extra_params=(rmse_train_per_iter, train_accuracy),
             labels=['K', 'RMSE for Train', 'Accuracy', 'Params'])

""" Keep the top 5 best """
top = 5

best_indices = np.array([_[-1] for _ in rmse_train_per_iter]).argsort()[:top]
copy_best_param_per_iter = copy.deepcopy([best_param_per_iter[i] for i in best_indices])
_, counts = np.unique([p['id'] for p in copy_best_param_per_iter], return_counts=True)

best_iter = best_indices[counts.argmax()]

# Copy the params because I dont want to accidently reset them...
copy_rmse_train = copy.deepcopy(rmse_train_per_iter)
copy_train_accuracy = copy.deepcopy(train_accuracy)

""" Plot the test and train RMSE """
plt.figure(figsize=(14,10))
plt.subplot(3,1,1)
plt.plot(copy_rmse_train[best_iter])
plt.title(f'Best Param is: {get_formatted_params(best_param_per_iter[best_iter], include_values=True)} \n\n with final'+
          f' test RMSE of {copy_rmse_train[best_iter][-1]}')
plt.xlabel('Epochs')
plt.ylabel('RMSE of best for Test and training')
plt.subplot(3,1,2)
plt.plot(copy_train_accuracy[best_iter])
plt.title(f'Best Param is: {get_formatted_params(best_param_per_iter[best_iter], include_values=True)} \n\n with final'+
          f' train accuracy of {copy_train_accuracy[best_iter][-1]}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy of best for training')

nn = NeuralNetLogReg(best_param_per_iter[best_iter]['struct'])
""" Train Neural Network """
nn.train(X_train, y_train, ftracep=True, wtracep=True)

prediction_y = nn.use(X_test)

plt.subplot(3, 1, 2)
plt.plot(range(len(X_test)), np.argmax(y_test, axis=1), 'o-', range(len(X_test)), prediction_y, 'o-')
plt.xlim(0, int(len(X_test)/2))
plt.legend(('Testing', 'Model'), loc='upper left')
plt.xlabel('$x$')
plt.ylabel('Actual and Predicted $f(x)$')
plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), prediction_y)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[classification_name[c] for c in classification_name],
                      title='Confusion matrix, without normalization')
plt.show()

from sklearn.metrics import precision_recall_fscore_support
metrics = precision_recall_fscore_support(np.argmax(y_test, axis=1), prediction_y, average='weighted')
plot_precision_recall_f1(metrics, ['Precision', 'Recall', 'f Score'])
plt.show()
