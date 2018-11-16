from logging import debug

import numpy as np


class Layer(object):
    """
    So the main goals of the layer is to take in a X or a Z and
    produce a Z[+1] (next layer's Z). It should describe the activation a.
    It should describe the activation rate (prime)
    """
    def __init__(self, size, name='placeholder', is_input=False, is_output=False,
                 weight_init_mode='glorot', activation='sigmoid', use_bias=False,
                 bias_default=1):
        """
        So this is the scheme.
        The layers should support bias, different ways of activating,
        and change how they behave based on if they are input or output.

        If input layer, then it should be able to do the example:
        X = (n_samples x n_features)
        w = (n_features x next Layer size)

        :param size: The size of w, a, and z
        :param name:
        :param is_input: Used for auto-normalizing
        :param is_output: Used to set the "next layer" size to 1 since it is output
        :param weight_init_mode: How are the weights going to be initialized
        :param use_bias: Should this layer use a bias?
        """
        self._is_output = is_output
        self._is_input = is_input
        self._weight_init_mode = weight_init_mode
        self.name = name

        if type(size) is tuple:
            size = size[0]

        self._bias_default = bias_default
        self._use_bias = use_bias
        self.size = size + 1 if use_bias else size
        self.w = None  # type: np.array
        self.w_gradient = None  # type: np.array
        self.a = None  # type: np.array
        self.z = None  # type: np.array
        self._activation = activation

        # Order check
        self._layer_is_built = False

    def build(self, next_layer_size=1, next_layer_using_bias=False):

        # The layer should not output the same size as the next layer if the next layer is including bias in its
        # size. So the current layer will output -1 the size, and corresponding weights accordingly.
        next_layer_size = next_layer_size - 1 if next_layer_using_bias else next_layer_size

        # Validate
        if next_layer_size != 1 and self._is_output:
            raise AttributeError(f'Layer {self.name} is defined as an '
                                 f'output layer. Way is the next layer not 1?')

        # Init w based on next layer size
        if self._weight_init_mode == 'random':
            # Initializes around a normal distribution
            self.w = np.random.randn((self.size, next_layer_size))
        elif self._weight_init_mode == 'middle':
            self.w = np.ones((self.size, next_layer_size))
        elif self._weight_init_mode == 'glorot':
            limit = np.sqrt(6 / sum((self.size, next_layer_size)))
            self.w = np.random.uniform(-limit, limit, (self.size, next_layer_size))
        else:
            self.w = np.zeros((self.size, next_layer_size))

        # Init other components relative to w
        self.a = np.zeros(self.w.shape)  # Current Layer inputs
        self.z = np.zeros(self.w.shape)  # Current Layer a @ w
        self.w_gradient = np.copy(self.w)  # Current Layer gradients

        # Fix w
        if self._is_output:
            debug(f'I am an {self.name}, so there is no w output. Setting this to None will prevent accidental'
                  f'usage.')
            self.w = None

    def forward(self, z: np.array):
        """
        z should be (n_samples x n_features) x (prev_layer_features x layer_size)
        This will be the input z for the next layer ie:

        z(2) = X(or Z(1))W(1)

        :param z:
        :return:
        """
        # So, the primary variables: z, and a get set
        self.z = z  # Sets the layers current z ie the input z
        if self._use_bias:
            self.z = np.hstack((self.z, np.full((self.z.shape[0], 1), fill_value=self._bias_default)))

        # Major change here, the layer decides if it need to do an activation
        if not self._is_input:
            self.a = self.activation(self.z)
        else:
            debug(f'I am {self.name} and I am an input layer. So I am not going to run an activation function')
            self.a = self.z

        if not self._is_output:
            return self.a @ self.w
        elif self._is_output and self._activation == 'softmax':
            return self.activation(self.a)
        else:
            return self.a

    def activation(self, z: np.array):
        if self._activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self._activation == 'softmax':
            """ Noted: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/ """
            shift_z = z - np.max(z)
            return np.exp(z) / np.sum(np.exp(z))
        else:
            return 0

    def activation_prime(self, z: np.array):
        # if self._use_bias:
        #     z = np.hstack((z, np.full((z.shape[0], 1), fill_value=self._bias_default)))

        if self._activation == 'sigmoid':
            return np.exp(-z) / ((1 + np.exp(-z)) ** 2)
        elif self._activation == 'softmax':
            return z * (1 - z)
        else:
            return 0
