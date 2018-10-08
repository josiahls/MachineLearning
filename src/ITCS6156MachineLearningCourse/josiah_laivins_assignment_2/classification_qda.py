import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# mu1 = [-1, -1]
# cov1 = np.eye(2)
#
# mu2 = [2, 3]
# cov2 = np.eye(2) * 3
#
# C1 = np.random.multivariate_normal(mu1, cov1, 50)
# C2 = np.random.multivariate_normal(mu2, cov2, 50)
#
# # big_x = np.vstack((C1, C2))
# big_x = np.array([
#     [1.0, 2.0],
#     [3.0, 5.0],
#     [2.1, 1.9]
# ])
#
#
# N = big_x.shape[0]
# big_t = np.ones(N).reshape(-1, 1)
# big_t[0] = 1
# big_t[1] = 2
# big_t[2] = 3


mu1 = [-1, -1]
cov1 = np.eye(2)

mu2 = [2,3]
cov2 = np.eye(2) * 3

C1 = np.random.multivariate_normal(mu1, cov1, 50)
C2 = np.random.multivariate_normal(mu2, cov2, 50)

plt.plot(C1[:, 0], C1[:, 1], 'or')
plt.plot(C2[:, 0], C2[:, 1], 'xb')

plt.xlim([-3, 6])
plt.ylim([-3, 7])
plt.show()

big_x = np.vstack((C1, C2))
big_t = np.ones(100)
big_t[:50] *= -1

# Train and Test data
N1 = C1.shape[0]
N2 = C2.shape[0]
N = N1 + N2


def gaussian(big_x: np.array, mean: np.array, sigma: np.array):
    """
    Assumption:

    big_x is (samples X features)
    mean is (features X 1)
    sigma is (feature X features)

    ...

    m is (features X 1) <- gaussian distribution

    :param big_x:
    :return:
    """

    # ## Single var: (1/sqrt(2*pi*sig^2)) * e ^ ( (-1/2) ((big_x - mean) / sig) ^ 2)
    # ## Multi-var: (1.0 / (2 * pi) ^ (features / 2) * (det(sigma)))

    # This is supposed to represent sigma ^ 2????
    det_sigma = sigma if sigma.shape[0] == 1 else np.linalg.det(sigma)
    # Sigma is used as a determinant on a fraction, so we take the inverse to keep it nice
    inverse_sigma = 1.0 / sigma if sigma.shape[0] == 1 else np.linalg.inv(sigma)
    # Sets the bottom distribution. Determines the spread.
    normalize_coeff = 1.0 / (np.sqrt(((2 * np.pi) ** sigma.shape[0]) * det_sigma))
    # Difference ie x - mean
    difference_between_mean = big_x - mean.T
    # Nat log exponent handles larger sigma's
    nat_log_exponent = (-0.5 * np.sum(np.dot(difference_between_mean, inverse_sigma) * difference_between_mean,
                                      axis=1))[:, np.newaxis]
    # Finally: normalize_coeff * e ** (nat_log_exponent)
    gaussian_distribution = normalize_coeff * np.exp(nat_log_exponent)
    return gaussian_distribution


def gaussian_parameters(big_x: np.array):
    """
    Should return:
    big_x: (samples X features)
    mean: (features X 1) <- multi-dim mean
    cov: (features X features) <- multi-dim sigma. Distribution matrix (covariance)


    :param big_x:
    :return:
    """
    return big_x, np.mean(big_x, axis=0).flatten(), np.cov(np.transpose(big_x))
    # return big_x, np.array([[2], [2]]).reshape(-1, 1), np.array([[1, 0], [0, 1]])


# ## Next we implement QDA: Quadratic Distribution Analysis
def QDA(sub_x: np.array, t_probability: float):
    _, mean, sigma = gaussian_parameters(sub_x)
    print(f'Custom Function output 1: {mean} \n {sigma}')
    # Get the probabilities for the positive class
    return gaussian(sub_x, mean, sigma) * t_probability

def old_QDA(big_x: np.array, mu, sigma, prior):
    sigma_inv = np.linalg.inv(sigma)
    diffv = big_x - mu
    return -0.5 * np.log(np.linalg.det(sigma)) \
        - 0.5 * np.sum(diffv @ sigma_inv * diffv, axis= 1) + np.log(prior)

# Train and Test data
N1 = C1.shape[0]
N2 = C2.shape[0]
N = N1 + N2

Xtrain = np.vstack((C1, C2))
Ttrain = np.ones(80)
Ttrain[:N1] *= -1


means, stds = np.mean(Xtrain, 0), np.std(Xtrain, 0)
Xs = (Xtrain - means) / stds

mu1 = np.mean(Xs[:N1], 0)
mu2 = np.mean(Xs[N1:], 0)

Sigma1 = np.cov(Xs[:N1].T)
Sigma2 = np.cov(Xs[N1:].T)


prior1 = N1 / N
prior2 = N2 / N

## now compute the discriminant function on test data

xs, ys = np.meshgrid(np.linspace(-3,6, 500), np.linspace(-3,7, 500))
Xtest = np.vstack((xs.flat, ys.flat)).T
XtestS = (Xtest-means)/stds

d1 = old_QDA(XtestS, mu1, Sigma1, prior1)
d2 = old_QDA(XtestS, mu2, Sigma2, prior2)


# print(f':Sigma1: {Sigma1} Mean1: {mu1}')
# print(f':Sigma2: {Sigma2} Mean2: {mu2}')
#
# d1 = QDA(Xs[:N1], prior1)
# d2 = QDA(Xs[N1:], prior2)

fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')
ax.plot_surface(xs, ys, d1.reshape(xs.shape), alpha=0.2)
ax.plot_surface(xs, ys, d2.reshape(xs.shape), alpha=0.4)
plt.title("QDA Discriminant Functions")

plt.figure(figsize=(6,6))
plt.contourf(xs, ys, (d1-d2 > 0).reshape(xs.shape))
plt.title("Decision Boundary")
plt.show()
# Plot generative distributions  p(x | Class=k)  starting with discriminant functions

fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')

prob1 = np.exp(d1.reshape(xs.shape) - 0.5*big_x.shape[1]*np.log(2*np.pi) - np.log(prior1))
prob2 = np.exp(d2.reshape(xs.shape) - 0.5*big_x.shape[1]*np.log(2*np.pi) - np.log(prior2))
ax.plot_surface(xs, ys, prob1, alpha=0.2)
ax.plot_surface(xs, ys, prob2, alpha=0.4)


plt.ylabel("QDA P(x|Class=k)\n from disc funcs", multialignment="center")
plt.show()

