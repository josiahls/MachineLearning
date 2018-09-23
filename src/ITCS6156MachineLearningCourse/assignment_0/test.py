import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas


import collections  # for checking iterable instance

# Super class for machine learning models

class BaseModel(ABC):
    """ Super class for ITCS Machine Learning Class"""

    @abstractmethod
    def train(self, X, T):
        pass

    @abstractmethod
    def use(self, X):
        pass


class LinearModel(BaseModel):
    """
        Abstract class for a linear model

        Attributes
        ==========
        w       ndarray
                weight vector/matrix
    """

    def __init__(self):
        """
            weight vector w is initialized as None
        """
        self.w = None

    def _check_matrix(self, mat, name):
        if len(mat.shape) != 2:
            raise ValueError(''.join(["Wrong matrix ", name]))

    # add a basis
    def add_ones(self, X):
        """
            add a column basis to X input matrix
        """
        self._check_matrix(X, 'X')
        return np.hstack((np.ones((X.shape[0], 1)), X))

    ####################################################
    #### abstract funcitons ############################
    @abstractmethod
    def train(self, X, T):
        """
            train linear model

            parameters
            -----------
            X     2d array
                  input data
            T     2d array
                  target labels
        """
        pass

    @abstractmethod
    def use(self, X):
        """
            apply the learned model to input X

            parameters
            ----------
            X     2d array
                  input data

        """
        pass

# LMS class
class LMS(LinearModel):
    """
        Lease Mean Squares. online learning algorithm

        attributes
        ==========
        w        nd.array
                 weight matrix
        alpha    float
                 learning rate
    """

    def __init__(self, alpha, k=-1, offset=0):
        LinearModel.__init__(self)
        self.alpha = alpha
        self.k = k
        self.w = []
        self.offset = offset

    # batch training by using train_step function
    def train(self, X, T):
        for x, t in zip(X, T):
            self.train_step(x, t)

    # train LMS model one step
    # here the x is 1d vector
    def train_step(self, x, t):
        N = x.reshape(-1, 1).shape[1]
        # # TODO: code for finding w
        x = np.hstack((np.ones((N, 1)), np.transpose(x.reshape(-1, 1))))

        if self.k == -1:
            self.k += 1
            self.w = np.array([np.random.rand(x.shape[1])])
        elif self.k != X.shape[0] - 1:
            #             self.w.append(0)
            print(str(self.w) + '\n\n' + str(x))
            y = np.transpose(self.w[self.k]) * x
            print('\nY: ' + str(y))

            self.w = np.concatenate(self.w, self.w[self.k] - self.alpha * (y - t) @x)
            self.k += 1

    # apply the current model to data X
    def use(self, X):
        N = X.shape[0]
        X1 = np.hstack((np.ones((N, 1)), X.reshape((X.shape[0], -1))))
        return X1 @ np.transpose(self.w)

X = np.array([[2,5],
              [6,2],
              [1,9],
              [4,5],
              [6,3],
              [7,4],
              [8,3]])
T = X[:,0, None] * 3 - 2 * X[:, 1, None] + 3

import IPython.display as ipd  # for display and clear_output
fig = plt.figure()

lms = LMS(0.02)

print('Targets:'+ str(T) + ' X: ' + str(X))
for x, t in zip(X, T):
    lms.train_step(x, t)
    plt.clf()
    plt.plot(lms.use(X))
    ipd.clear_output(wait=True)
    ipd.display(fig)
ipd.clear_output(wait=True)

plt.plot(T, label='Ground Truth')
plt.legend()