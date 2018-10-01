import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0,10, 101)
T = 2 * X  + 4+ np.random.rand(101) * 5  # Is this y?????

def data_scatter(k=101):
    plt.plot(T[:k], '.')
    plt.xticks(range(0, 101, 20)[:k], range(0, 11, 2)[:k])

#mean = np.mean(T)
#data_scatter()
#plt.plot([0, 100],[mean, mean], 'r-')
#plt.show()


def least_squares(sample_x: list, sample_y: list):
    # Goal is yi = b0_hat + b1_hat * xi

    # First we need b1_hat
    Sxx = sum([(x - sum(sample_x) / len(sample_x)) ** 2 for x in sample_x])
    Sxy = sum([(x - sum(sample_x) / len(sample_x)) * (y - sum(sample_y) / len(sample_y)) for x, y in
               zip(sample_x, sample_y)])

    b1_hat = Sxy / Sxx
    # Then we get b0_hat
    b0_hat = sum(sample_y) / len(sample_y) - b1_hat * (sum(sample_x) / len(sample_x))

    print(f'Sxx: {Sxx} Sxy: {Sxy}')
    print(f'b1_hat is: {b1_hat} and b0_hat is {b0_hat}')
    # And so the resulting equation can be
    # y = b0_hat + b1_hat * x where x is the 'time period' and y is the target
    return b1_hat, b0_hat


b1_hat, b0_hat = least_squares(X, T)
regression_y = [b0_hat + b1_hat * x for x in X]
print(regression_y)

data_scatter()
plt.plot([0, 100],[regression_y[0], regression_y[-1]], 'r-')
plt.show()

import IPython.display as ipd  # for display and clear_output

# initial weights with random values
w = np.random.rand(X.shape[0])  # Random weights
w[0] = 1

# learning rate
alpha = 0.1

fig = plt.figure()

# sequential learning
for k in range(X.shape[0]):
    # TODO: online update of weights
    if k != X.shape[0] - 1:
        w[k + 1] = w[k] - alpha * (w[k] * X[k] - T[k]) * X[k]

    plt.clf()

    plt.plot([0, 100], [w[0], w[k]], 'r-')
    data_scatter(k + 1)

    ipd.clear_output(wait=True)
    ipd.display(fig)
ipd.clear_output(wait=True)

