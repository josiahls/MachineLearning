from LeeNeuralNet import NeuralNet
import numpy as np

class NeuralNetLogReg(NeuralNet):
    """ Nonlinear Logistic Regression
    """

    # if you think, you need additional items to initialize here,
    # add your code for it here
    def __init__(self, nunits, is_image=False, standardize_target=False):
        super().__init__(nunits)
        self.is_image = is_image
        self.stdTarget = standardize_target

    # Looking at the final summary or comparison table in lecture note,
    # add your codes for forward pass for logistic regression
    def forward(self, X):
        y, z = NeuralNet.forward(self, X)
        # Softmax

        # Used https://stackoverflow.com/questions/43290138/softmax-function-of-a-numpy-array-by-row
        # Last answer for softmax.
        return self.softmax(y, axis=1), z

    def softmax(self, a, axis=None):
        """
        Computes exp(a)/sumexp(a); relies on scipy logsumexp implementation.
        :param a: ndarray/tensor
        :param axis: axis to sum over; default (None) sums over everything
        """
        from scipy.special import logsumexp
        lse = logsumexp(a, axis=axis)  # this reduces along axis
        if axis is not None:
            lse = np.expand_dims(lse, axis)  # restore that axis for subtraction
        return np.exp(a - lse)

    # This is the error function that we want to minimize
    # what was it? take a look at the lecture note to fill in
    def _objectf(self, T, Y, wpenalty):
        return NeuralNet._objectf(self, T, Y, wpenalty)

    # you must reuse the NeuralNet train since you already modified
    # the objective or error function (maybe both),
    # you do not have many to change here.
    # MAKE SURE convert a vector label T to indicator matrix and
    # feed that for training
    def train(self, X, T, ftracep=True, wtracep=True):
        return NeuralNet.train(self=self, X=X, T=T, ftracep=ftracep, wtracep=wtracep)

    # going through forward pass, you will have the probabilities for each label
    # now, you can use argmax to find class labels
    # return both label and probabilities
    def use(self, X, retZ=False):
        return np.argmax(NeuralNet.use(self, X, retZ), axis=1)
