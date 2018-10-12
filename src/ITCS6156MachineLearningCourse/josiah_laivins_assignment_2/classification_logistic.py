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
big_t = np.zeros((100, 2))
big_t[:50, 0] = 1
big_t[50:, 1] = 1

# Train and Test data
N1 = C1.shape[0]
N2 = C2.shape[0]
N = N1 + N2


def softmax(z):
    if not isinstance(z, np.ndarray):
        z = np.asarray(z)
    f = np.exp(z)
    return f / (np.sum(f, axis= 1, keepdims=True)) if len(z.shape) == 2 else np.sum(f)

def g(big_x, w):
    return softmax(big_x @ w)

# w = np.random.rand((1, 1))

def get_logistic(big_x: np.array, big_t: np.array):
    D = 2
    K = 2
    w = np.random.rand(D+1, K)
    iterations = 1000
    alpha = 0.1

    bais_big_x = np.hstack((np.ones((N, 1)), big_x))
    likelihood = []

    for step in range(iterations):
        ys = g(bais_big_x, w)
        w += alpha * bais_big_x.T @ (big_t - ys)

# Train and Test data
means, stds = np.mean(big_x, 0), np.std(big_x, 0)
Xs = (big_x - means) / stds



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



get_logistic(Xs, big_t)
Xfinal = np.hstack(())
Yl = np.argmax(Y, 1)
Tl = np.argmax(Ttest, 1)

plt.plot(Tl)
plt.plot(Yl)

print("Accuracy: ", 100 - np.mean(np.abs(Tl - Yl)) * 100, "%")

# show me the boundary

x = np.linspace(-3, 6, 1000)
y = np.linspace(-3, 7, 1000)

xs, ys = np.meshgrid(x, y)

X = np.vstack((xs.flat, ys.flat)).T
X1 = np.hstack((np.ones((X.shape[0], 1)), X))

Y = g(X1, w)
zs = np.argmax(Y, 1)

plt.figure(figsize=(6,6))
plt.contourf(xs, ys, zs.reshape(xs.shape))
plt.title("Decision Boundary")

plt.plot(C1[:, 0], C1[:, 1], 'or')
plt.plot(C2[:, 0], C2[:, 1], 'xb')

