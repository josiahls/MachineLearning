# from nn import NeuralNet
import sys
from logging import *
from typing import List
import numpy as np
from numpy.core.multiarray import ndarray
from tqdm import tqdm
from Layer import Layer

basicConfig(stream=sys.stderr, level=INFO)


# noinspection PyPep8Naming
class NeuralNetLogReg(object):
    """ Nonlinear Logistic Regression

    Large amounts of this implementation can be credited to:
    https://github.com/stephencwelch/Neural-Networks-Demystified
    """

    def __init__(self, X_train=np.array([]), Y_train=np.array([]), X_test=np.array([]), Y_test=np.array([])):
        self.layers = []  # type: List[Layer]
        self.log_rmse_train = []
        self.log_rmse_test = []
        self.cost_log = []  # Also J
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    # Looking at the final summary or comparison table in lecture note,
    # add your codes for forward pass for logistic regression
    def forward(self, X):
        # Start off with X serving as the 'z', then...
        # Prep the input layer. It should:
        # Set a and z to x, and have the next z as x @ w
        next_z = self.layers[0].forward(X)

        for i in range(1, len(self.layers)):  # This is hidden.
            # ... go through the rest of the layers propagating 'z' through them
            next_z = self.layers[i].forward(next_z)

        return next_z  # Final Result is the last layer's activation

    # This is the error function that we want to minimize
    # what was it? take a look at the lecture note to fill in
    def _objectf(self, T, Y, wpenalty):
        pass

    def cost_function(self, x_scaled, actual_y):
        """
        Also known as J.
        Does: sum((1/2) * (actual - predicted)^2)

        The goal of the back prop will be to make this as small as possible.

        :param x_scaled:
        :param actual_y:
        :return:
        """
        """ FORMULA 3: J = sum((1/2) * (y - y_hat)^2) """
        """ Forward Propagate """
        predicted_y = self.forward(x_scaled)
        return np.sum((actual_y - predicted_y) ** 2) * 0.5

    # noinspection PyUnresolvedReferences
    def cost_function_prime(self, x_scaled, actual_y):
        """
        Explaination:

        So 2 important variables:
        back_propagating_error:
        - Based on how different the predicted y is and the actual y
        - Based on the whether the current layer's activation is going up or down (and by how much)
        - Distribute the error across them
        - The final result is a set of errors for each weight. Flat weights will be zero and say that
        there is not affect, while weight that are in the middle of changing a lot will have more
        error applied to them.

        gradient:
        - Based on what the layer is outputting
        - Based on how much error each neuron is producing
        - The current activation value will the error considered is being outputted


        :param x_scaled:
        :param actual_y:
        :return:
        """

        """ FORMULA 5: 

        dJ / dW(i) = -(y-predicted_y) * gradient(i+1) @ dZ(i+1) / dW(i)

        OR

        delta[i+1] = -(y - predicted_y) * costFunctionPrime(z[i+1])
        dJ / dW(i) = a(i).T @ delta[i+1]

        As a Note:
        a = current layer input
        z = a @ w current layer output

        """
        predicted_y = self.forward(x_scaled)

        delta: int = 0
        # Move backwards, excluding the output layer
        for i in range(len(self.layers) - 1, 0, -1):  # Layers last - 1 -> 1
            # We this is the back prop error of this current layer based on the inputs, the weight, and
            # the current output
            z = self.layers[i].z  # if not self.layers[i]._use_bias else self.layers[i].z[:, :-1]
            if type(delta) is int:
                delta = -1 * (actual_y - predicted_y) * self.layers[i].activation_prime(z)
            else:
                delta = (delta @ self.layers[i].w.T) * self.layers[i].activation_prime(z)
                delta = delta if not self.layers[i]._use_bias else delta[:, :-1]
            gradient = self.layers[i - 1].a.T @ delta

            self.layers[i - 1].w_gradient = gradient
            # debug(f'Gradient of layer {i} is \n {self.layers[i].gradient_of_w }')

        parameters = np.array([])
        for i in range(len(self.layers) - 1):
            parameters = np.concatenate((parameters, self.layers[i].w_gradient.flatten()))

        return parameters

    def _build(self):
        for i in range(len(self.layers) - 1):
            self.layers[i].build(self.layers[i + 1].size, self.layers[i + 1]._use_bias)
        self.layers[-1].build()

    # you must reuse the NeuralNet train since you already modified
    # the objective or error function (maybe both),
    # you do not have many to change here.
    # MAKE SURE convert a vector label T to indicator matrix and
    # feed that for training
    def train(self, X, Y, epochs=100):
        """
        Assuming that X and Y are properly normalized

        Train's goal is the following:
        - build the layers in the network
        - call minimize function

        :param X:
        :param Y:
        :param epochs:
        :return:
        """
        self.X_train = X
        self.Y_train = Y
        """ Build the network """
        self._build()
        """ Do back prop """
        params_init = self.get_parameters_as_weights()

        self.cost_log = []  # Set the cost log to init state
        # options = {'maxiter': epochs, 'disp': True}
        # results = minimize(self.cost_function_wrapper, params_init,
        #                    jac=True, method='BFGS', args=(X, Y),
        #                    options=options, callback=self.call_back)
        self.minimize(self.cost_function_wrapper(params_init, X, Y), X, Y, epochs=epochs)

        # self.set_rmse(self.X, self.Y)
        # debug(f'Result: \n{self.forward(self.X_train)}\n and RMSE is {self.log_rmse_train}')

    def minimize(self, grad, x, y, epochs=100, alpha=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08):
        """
        This adam optimizer is credited to UNCC course:
        201880-ITCS-4152-001:ITCS-5152-001-XLSE8201880_CombinedModules Fall 2018
        """

        m0 = np.zeros(len(grad))  # Initialize first moment vector
        v0 = np.zeros(len(grad))  # Initialize second moment vector
        t = 0.0

        self.cost_log = []  # For visualization
        mt = m0
        vt = v0

        for _ in tqdm(range(epochs)):
            t += 1
            grads = self.compute_gradients(x_scaled=x, scaled_y=y)
            mt = beta1 * mt + (1 - beta1) * grads
            vt = beta2 * vt + (1 - beta2) * grads ** 2
            mt_hat = mt / (1 - beta1 ** t)
            vt_hat = vt / (1 - beta2 ** t)

            params = self.get_parameters_as_weights()
            new_params = params - alpha * mt_hat / (np.sqrt(vt_hat) + epsilon)
            self.set_parameters_as_weights(new_params)

            # Log each epoch
            self.cost_log.append(self.cost_function(x_scaled=x, actual_y=y))
            self.set_rmse(x, y, dataset_type='train')
            self.set_rmse(self.X_test, self.Y_test, dataset_type='test')

    def call_back(self, params):
        # print(f' Iterating ')
        self.set_parameters_as_weights(params)
        self.cost_log.append(self.cost_function(x_scaled=self.X_train, actual_y=self.X_train))

    def cost_function_wrapper(self, params, x_scaled, y_scaled):
        # print(f'Iteration Size: {len(self.cost_log)}')
        self.set_parameters_as_weights(params)
        # cost = self.cost_function(x_scaled, y_scaled)
        grad: ndarray = self.compute_gradients(x_scaled, y_scaled)

        # print(f'Cost: {cost}')
        return grad

    def predict(self, x):
        predicted_y = self.forward(x)
        return np.argmax(predicted_y, axis=1)

    # going through forward pass, you will have the probabilities for each label
    # now, you can use argmax to find class labels
    # return both label and probabilities
    def use(self, X):
        return self.predict(X)

    def set_rmse(self, x_scaled, actual_y, dataset_type='train'):
        predicted_y = self.forward(x_scaled)
        rmse = np.sqrt(np.mean((np.array(predicted_y) - actual_y) ** 2))

        if dataset_type == 'train':
            self.log_rmse_train.append(round(rmse, 3))
        else:
            self.log_rmse_test.append(round(rmse, 3))

    def get_parameters_as_weights(self) -> np.array:
        """
        Unravels all the weights in the network into a single easy line.

        :return:
        """
        # So here is the scheme:
        # layer.w.flatten() -> make an easy flat array
        # layer.w <- np.reshape([number of elements], layer.w.shape) set the param back to the array

        parameters = np.array([])
        for i in range(len(self.layers) - 1):
            parameters = np.concatenate((parameters, self.layers[i].w.flatten()))

        return parameters

    def set_parameters_as_weights(self, params: np.array):
        """
        Gets the new set of weights as a single vector line.

        :param params: 1 x N size vector
        :return:
        """

        param_index = 0
        # For each layer
        for i in range(len(self.layers) - 1):
            self.layers[i].w = np.reshape(params[param_index:param_index + self.layers[i].w.size],
                                          self.layers[i].w.shape)
            param_index += self.layers[i].w.size

    def compute_gradients(self, x_scaled, scaled_y):
        return self.cost_function_prime(x_scaled, scaled_y)
