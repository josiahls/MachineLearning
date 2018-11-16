import numpy as np


# noinspection PyPep8Naming
class Standardizer:
    """ class version of standardization """
    def __init__(self, X: np.array, explore=False, is_image=False):
        if not is_image:
            self._mu = np.mean(X, axis=0)
            self._sigma = np.std(X, axis=0)
        else:
            self._mu = 0
            self._sigma = 255

        if explore:
            print ("mean: ", self._mu)
            print ("sigma: ", self._sigma)
            print ("min: ", np.min(X, axis=0))
            print ("max: ", np.max(X, axis=0))

    def set_sigma(self, s):
        self._sigma[:] = s

    def standardize(self, X: np.array):
        return np.nan_to_num((X - self._mu) / self._sigma)

    def unstandardize(self, X: np.array):
        return (X * self._sigma) + self._mu