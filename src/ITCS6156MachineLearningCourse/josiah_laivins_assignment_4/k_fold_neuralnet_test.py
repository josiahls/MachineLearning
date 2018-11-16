import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm_notebook as tqdm

from Layer import Layer
from NeuralNetwork import NeuralNetLogReg
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
train = pd.read_csv('./data/fashionmnist/fashion-mnist_train.csv', nrows=300)
test = pd.read_csv('./data/fashionmnist/fashion-mnist_test.csv', nrows=300)

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

standardizer_x = Standardizer(X)
stand_X = standardizer_x.standardize(X)
encoder = OneHotEncoder()
encoder.fit(Y)
stand_Y = encoder.transform(Y).toarray()


""" Set up K-Fold """
k_folds = 2

# Logs for K and the params to test
train_params = [{'n_hidden_layers': 1, 'layer_sizes': [32], 'epochs': 50, 'input_bias':False, 'hidden_bias': [False],
                 'activation_type': ['sigmoid'], 'output_type': None},
                {'n_hidden_layers': 1, 'layer_sizes': [32], 'epochs': 50, 'input_bias':True, 'hidden_bias': [True],
                 'activation_type': ['sigmoid'], 'output_type': 'softmax'},
                # {'n_hidden_layers': 1, 'layer_sizes': [16], 'epochs': 500, 'input_bias':True, 'hidden_bias': [True],
                #  'activation_type': ['sigmoid']},
                ]
rmse_test_per_iter = []
rmse_train_per_iter = []
cost_log_per_iter = []
best_param_per_iter = []

""" Add unique ids for each param to be identified by """
for i in range(len(train_params)):
    train_params[i]['id'] = i

# For each K
for k in tqdm(range(k_folds), bar_format='r_bar'):
    X_train, X_test, y_train, y_test = train_test_split(stand_X, stand_Y, test_size=0.33)

    # Test each param
    for param in train_params:
        """ Build Neural Net """
        nn = NeuralNetLogReg(X_train, y_train, X_test, y_test)
        nn.add_layer(Layer(X_train[0].shape, is_input=True, name='Input Layer', use_bias=True))
        # Based on the params, build the network
        for i in range(param['n_hidden_layers']):
            nn.add_layer(Layer(param['layer_sizes'][i], name=f'Hidden Layer {i}',
                               use_bias=param['hidden_bias'][i],
                               activation=param['activation_type'][i]))
        nn.add_layer(Layer(y_train.shape[1], is_output=True, name='Output Layer', activation=param['output_type']))

        """ Train Neural Network """
        nn.train(X_train, y_train, epochs=param['epochs'])

        # Set test rmse local or res
        rmse_test_per_iter.append(nn.log_rmse_test)
        rmse_train_per_iter.append(nn.log_rmse_train)
        cost_log_per_iter.append(nn.cost_log)
        best_param_per_iter.append(param)

""" Show the parameters and their performance """
plot_k_folds(k_folds, best_param_per_iter, rmse_test_per_iter, rmse_train_per_iter, cost_log_per_iter)
plt.show()

""" Keep the top 5 best """
import copy

top = 2

best_indices = np.array([_[-1] for _ in rmse_test_per_iter]).argsort()[:top]
copy_best_param_per_iter = copy.deepcopy([best_param_per_iter[i] for i in best_indices])
_, counts = np.unique([p['id'] for p in copy_best_param_per_iter], return_counts=True)

best_iter = best_indices[counts.argmax()]

# Copy the params because I dont want to accidently reset them...
copy_rmse_test = rmse_test_per_iter[best_iter]
copy_rmse_train = rmse_train_per_iter[best_iter]
copy_cost_log = cost_log_per_iter[best_iter]
copy_best_param = best_param_per_iter[best_iter]

""" Plot the test and train RMSE """
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.plot(copy_rmse_test)
plt.title(f'Best Param is: {get_formatted_params(best_param_per_iter[best_iter], include_values=True)} \n\n with final'+
          f' test RMSE of {rmse_test_per_iter[best_iter][-1]}')
plt.xlabel('Epochs')
plt.ylabel('RMSE of best for Test and training')
plt.plot(copy_rmse_train)
plt.legend(('Test','Train'),loc='upper left')

plt.subplot(3,1,2)
plt.plot(copy_cost_log)
plt.xlabel('Epochs')
plt.ylabel('Cost of best')

plt.show()