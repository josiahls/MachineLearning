import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from Layer import Layer
from NeuralNetwork import NeuralNetLogReg

# %matplotlib inline
from util import plot_confusion_matrix, plot_precision_recall_f1

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
norm_X = standardizer_x.standardize(X)
encoder = OneHotEncoder()
encoder.fit(Y)
norm_Y = encoder.transform(Y).toarray()


""" Build Neural Net """
nn = NeuralNetLogReg(X_test=norm_X, Y_test=norm_Y)
nn.add_layer(Layer(norm_X[0].shape, is_input=True, name='Input Layer', use_bias=True))
nn.add_layer(Layer(32, name='Hidden Layer'))
nn.add_layer(Layer(len(np.unique(Y)), is_output=True, name='Output Layer'))

""" Train Neural Network """
nn.train(norm_X, norm_Y, epochs=200)
# print(f'RMSE train: {nn.log_rmse_train[-1]} Last cost: {nn.cost_log[-1]}')

plt.plot(nn.cost_log)
plt.show()

""" Use Neural Network """
y_predict = nn.use(norm_X)
# Compute confusion matrix
cnf_matrix = confusion_matrix(Y, y_predict)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[classification_name[c] for c in classification_name],
                      title='Confusion matrix, without normalization')
plt.show()


from sklearn.metrics import precision_recall_fscore_support
metrics = precision_recall_fscore_support(Y, y_predict, average='weighted')
plot_precision_recall_f1(metrics, ['Precision', 'Recall', 'f Score'])
plt.show()