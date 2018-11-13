from Layer import Layer
from NeuralNetwork import NeuralNetLogReg
# from NeuralNet2 import NeuralNetwork2
from NeuralNetBench import NeuralNet
import unittest
import numpy as np
import cv2
import matplotlib.pyplot as plt

class TestNeuralNetwork(unittest.TestCase):

    def test_derivative(self):

        def f(x):
            return x ** 2

        epsilon = 1e-4
        x = 1.5
        gradient = (f(x+epsilon) - f(x-epsilon))/(2*epsilon)

        self.assertAlmostEqual(2*x, gradient, delta=0.01)

    def test_gradient_derivative(self):

        # TODO Need to rework: [link](https://www.youtube.com/watch?v=pHMzNW8Agq4&pbjreload=10)
        # The reason he unravels the weights is so they are just one giant
        # long vector
            # Set up training data (feature set)

        """ Set Up NN """
        x = np.array([
            [3, 5],  # (time sleeping, studying)
            [5, 1],
            [10, 2]
        ], dtype=float)
        # Set up targets to train toward (classes)
        y = np.array(([75], [82], [93]), dtype=float)

        x = x / np.amax(x)
        y = y / 100
        epsilon = 1e-4
        # nn = NeuralNetwork2()
        nn = NeuralNetLogReg(2)
        # nn = NeuralNet()
        nn.add_layer(Layer(2, name='Input Layer', is_input=True))
        nn.add_layer(Layer(3, name='Hidden Layer'))
        nn.add_layer(Layer(1, name='Output Layer', is_output=True))
        nn._build()
        # nn.add_layer(2, 3, 'Input Layer')  # Input Layer
        # nn.add_layer(3, 1, 'Hidden Layer')  # Hidden Layer
        # nn.add_layer(1, 1, 'Output Layer')  # Output Layer

 #        nn.set_parameters_as_weights(np.array([ 0.47102342,  1.6876814,  -0.52526275,  1.72416179,  0.98861168,  0.2165734,
 # -0.59859513,  0.03984992,  0.213429]))

        """ Get current weights """
        init_params = nn.get_parameters_as_weights()
        numGrad = np.zeros(init_params.shape)
        perturb = np.zeros(init_params.shape)

        for p in range(len(init_params)):
            perturb[p] = epsilon

            nn.set_parameters_as_weights(init_params + perturb)
            loss2 = nn.cost_function(x, y)

            nn.set_parameters_as_weights(init_params - perturb)
            loss1 = nn.cost_function(x, y)

            numGrad[p] = (loss2 - loss1) / (2*epsilon)

            perturb[p] = 0

        nn.set_parameters_as_weights(init_params)
        grad = nn.compute_gradients(x, y)

        print(f'Numerical Gradient: {numGrad}')
        print(f'\n\n {grad}')

        print(f'Norm: {np.linalg.norm(grad-numGrad) / np.linalg.norm(grad+numGrad)}')
        nn.train(x, y, 50000)

        predictions = nn.predict(np.array(x))
        # Goal is [~90, ~70]
        print(f'Predictions: {predictions}')

        plt.plot(nn.cost_log)
        plt.show()

if __name__ == '__main__':
    unittest.main()