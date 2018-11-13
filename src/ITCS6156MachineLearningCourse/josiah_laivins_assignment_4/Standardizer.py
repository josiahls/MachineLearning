import numpy as np


# noinspection PyPep8Naming
class Standardizer:
    """ class version of standardization """
    def __init__(self, X: np.array, explore=False):
        self._mu = np.mean(X, axis=1).reshape(-1, 1)
        self._sigma = np.std(X, axis=1).reshape(-1, 1)
        if explore:
            print ("mean: ", self._mu)
            print ("sigma: ", self._sigma)
            print ("min: ", np.min(X, axis=1))
            print ("max: ", np.max(X, axis=1))

    def set_sigma(self, s):
        self._sigma[:] = s

    def standardize(self, X: np.array):
        return (X - self._mu) / self._sigma

    def unstandardize(self, X: np.array):
        return (X * self._sigma) + self._mu