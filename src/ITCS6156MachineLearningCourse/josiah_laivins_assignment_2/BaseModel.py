import numpy as np
from abc import *


# Super class for machine learning models

class BaseModel(ABC):
    """ Super class for ITCS Machine Learning Class"""

    @abstractmethod
    def train(self, X, T):
        pass

    @abstractmethod
    def use(self, X):
        pass


class Classifier(BaseModel):
    """
        Abstract class for classification

        Attributes
        ==========
        meanX       ndarray
                    mean of inputs (from standardization)
        stdX        ndarray
                    standard deviation of inputs (standardization)
    """

    def __init__(self):
        self.meanX = None
        self.stdX = None

    def normalize(self, X, reset_fields=False):
        """ standardize the input X """

        if not isinstance(X, np.ndarray):
            X = np.asanyarray(X)

        if reset_fields:
            self.meanX = np.mean(X, 0)
            self.stdX = np.std(X, 0)

        Xs = (X - self.meanX) / self.stdX
        return Xs

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
        pass

    @abstractmethod
    def use(self, X):
        pass
