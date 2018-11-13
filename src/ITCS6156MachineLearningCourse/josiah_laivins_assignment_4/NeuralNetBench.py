import numpy as np
from scipy.optimize import minimize
import sys
from logging import *
from tqdm import tqdm
import matplotlib.pyplot as plt

basicConfig(stream=sys.stderr, level=DEBUG)


# noinspection PyMethodMayBeStatic
class Layer:
    def __init__(self, size: int = 1, next_layer_size: int = 1, name: str = 'placeholder'):
        """
        Notes:

        self.w dims: (previous layer size x next layer size)
        self.gradient_of_w: same dim as w, but instead for each w
                            shows the amount of change that each weight
                            has / contributed to the predicted y and
                            finally the error.

        :param size: Also can be viewed as current feature set.
            If this is the input layer, then this is the n_features
            of the input data.
        """
        self.name = name
        self.w = self._get_weight_init((size, next_layer_size))  # Current Layer weights
        self.a = np.zeros(self.w.shape)  # Current Layer inputs
        self.z = np.zeros(self.w.shape)  # Current Layer a @ w
        self.gradient_of_w = np.copy(self.w)  # Current Layer gradients

    def _get_weight_init(self, w_shape: tuple, mode: str = 'glorot'):
        """
        Initialize the weight matrix by n_feature size x next_layer size.

        :param w_shape: n_features(or current layer size) x (next Layer size)
        :param mode: How the weights are being initialized
        :return: An initialized weight matrix.

        If input layer, then it should be able to do the example:
        X = (n_samples x n_features)
        w = (n_features x next Layer size)
        """
        if mode == 'random':
            # Inits around a normal distribution
            return np.random.randn(*w_shape)
        elif mode == 'middle':
            return np.ones(*w_shape)
        elif mode == 'glorot':
            limit = np.sqrt(6 / sum(w_shape))
            return np.random.uniform(-limit, limit, w_shape)
        else:
            return np.zeros(*w_shape)

    def get_forward_output(self, z: np.array):
        """
        Being: (n_samples x n_features) dot (prev layer features x layer size)
        This is rep of stephen's 1st formula z(2) = XW(1)
        Meaning, the next layer's values are are the dot product of the current layer's weights

        :param z: Is prev_z @ prev_w
        :return: Is the value that will server as the next layer's 'X'
        """
        # debug(f'{self.name} doing forward output using: z {z.shape} and w {self.w.shape}')
        # We get the new z given the set of weights
        """ FORMULA 1: (z = XW) """
        self.z = z
        # We perform an activation function on it
        self.a = self.activation_sigmoid(self.z)  # Save the outputs of the current layer

        return self.a @ self.w  # Return the outputs

    def activation_sigmoid(self, z: np.array):
        """ Applies a sigmoid activation function """
        """ FORMULA 2: (a = f(z)) """
        return 1 / (1 + np.exp(-z))

    def activation_sigmoid_prime(self, z: np.array):
        """ Applies a sigmoid activation function derivative """
        # Note this is for a larger equation belonging to the NN
        """ FORMULA 4: f'(z) """
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


# noinspection PyMethodMayBeStatic
class NeuralNet(object):

    def __init__(self):
        self.layers = []
        self.RMSE = []
        self.cost_log = []  # Also known as J
        self.max_y = 0.0
        self.max_x = 0.0

    def add_layer(self, size, next_layer_size, name):
        self.layers.append(Layer(size, next_layer_size, name))

    def train(self, x: np.array, y: np.array, epochs=1):
        """
        train does the following:

        - Scale inputs
        - For each layer - 1
            - next_z = layer.forward(next_z) # will do (X@w) and activation

        :param x:
        :param y:
        :return:
        """
        # Note this is not going to be ok if the values are negative
        # consider using X - mean / std(X)
        """ Normalize """
        self.x_scaled = x
        self.y_scaled = y

        """ Do Backprop """
        params_init = self.get_parameters_as_weights()

        self.cost_log = []
        # options = {'maxiter': epochs, 'disp': True}
        # results = minimize(self.cost_function_wrapper, params_init,
        #                    jac=True, method='BFGS', args=(x, y),
        #                    options=options, callback=self.call_back)
        self.minimize(*self.cost_function_wrapper(params_init, x, y), x, y)

        # The result is n_sample x target dim prediction.
        # These by default will look very bad
        self.get_rmse(self.x_scaled, self.y_scaled)
        # debug(f'Result: \n{self.forward(x_scaled)}\n and RMSE is {self.RMSE}')
        # self.set_parameters_as_weights(results.x)
        # self.optimization_results = results

        plt.plot(self.cost_log)
        plt.show()

    def minimize(self, cost, grad, x, y):
        num_iterations = 15000
        alpha = 1e-4
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08

        m0 = np.zeros(len(grad))  # Initialize first moment vector
        v0 = np.zeros(len(grad))  # Initialize second moment vector
        t = 0.0

        self.cost_log = []  # For visualization
        mt = m0
        vt = v0

        for i in tqdm(range(num_iterations)):
            t += 1
            grads = self.compute_gradients(x_scaled=x, scaled_y=y)
            mt = beta1 * mt + (1 - beta1) * grads
            vt = beta2 * vt + (1 - beta2) * grads ** 2
            mt_hat = mt / (1 - beta1 ** t)
            vt_hat = vt / (1 - beta2 ** t)

            params = self.get_parameters_as_weights()
            new_params = params - alpha * mt_hat / (np.sqrt(vt_hat) + epsilon)
            self.set_parameters_as_weights(new_params)

            self.cost_log.append(self.cost_function(x_scaled=x, actual_y=y))

    def predict(self, x):
        predicted_y = self.forward(x)
        # print(f'Predicting: {predicted_y}')
        return predicted_y

    def call_back(self, params):
        # print(f' Iterating ')
        self.set_parameters_as_weights(params)
        self.cost_log.append(self.cost_function(x_scaled=self.x_scaled, actual_y=self.y_scaled))

    def cost_function_wrapper(self, params, x_scaled, y_scaled):
        print(f'Iteration Size: {len(self.cost_log)}')
        self.set_parameters_as_weights(params)
        cost = self.cost_function(x_scaled, y_scaled)
        grad = self.compute_gradients(x_scaled, y_scaled)
        return cost, grad

    def forward(self, x_scaled) -> np.array:
        # Start off with X serving as the 'z', then...
        # Prep the input layer. It should:
        # Set a and z to x, and have the next z as x @ w
        self.layers[0].get_forward_output(x_scaled)
        self.layers[0].a = x_scaled
        next_z = self.layers[0].a @ self.layers[0].w

        for i in range(1, len(self.layers)):  # This is hidden.
            # ... go through the rest of the layers propagating 'z' through them
            next_z = self.layers[i].get_forward_output(next_z)

        predicted_y = self.layers[-1].a
        return predicted_y

    def cost_function(self, x_scaled, actual_y):
        """
        Also known as J.
        Does: sum((1/2) * (actual - predicted)^2)

        The goal of the back prop will be to make this as small as possible.

        :param predicted_y:
        :param actual_y:
        :return:
        """
        """ FORMULA 3: J = sum((1/2) * (y - y_hat)^2) """
        """ Forward Propagate """
        predicted_y = self.forward(x_scaled)
        return np.sum((actual_y - predicted_y) ** 2) * 0.5

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


        :param predicted_y:
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

        # Kind note to myself
        # FUCKING LEAVE THE OUTPUT LAYER ALONE
        # GODDAMNIT
        predicted_y = self.forward(x_scaled)

        delta: int = 0
        # Move backwards, excluding the output layer
        for i in range(len(self.layers) - 1, 0, -1):  # Layers last - 1 -> 1
            # We this is the back prop error of this current layer based on the inputs, the weight, and
            # the current output
            if type(delta) is int:
                delta = -1 * (actual_y - predicted_y) * self.layers[i].activation_sigmoid_prime(self.layers[i].z)
            else:
                delta = (delta @ self.layers[i].w.T) * self.layers[i].activation_sigmoid_prime(self.layers[i].z)

            gradient = self.layers[i - 1].a.T @ delta
            self.layers[i - 1].gradient_of_w = gradient
            # debug(f'Gradient of layer {i} is \n {self.layers[i].gradient_of_w }')

        parameters = np.array([])
        for i in range(len(self.layers) - 1):
            parameters = np.concatenate((parameters, self.layers[i].gradient_of_w.flatten()))

        return parameters

    def get_rmse(self, x_scaled, actual_y):
        predicted_y = self.forward(x_scaled)
        rmse = np.sqrt(np.mean((np.array(predicted_y) - actual_y) ** 2))
        self.RMSE.append(round(rmse, 3))

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

