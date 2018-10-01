import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as copy

# Example data
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

X = np.vstack((C1, C2))
N = X.shape[0]
T = np.ones(N)
T[:50] *= -1

maxiter = 4000
alpha = 0.01

w = np.zeros(1)
w_pocket = copy(w)
last_lowest_error = 1

plt.plot(T)
plt.show()

def compare(X, T, w, wp):
    y = np.sign(X @ w)
    yp = np.sign(X @ wp)

    return 1 if np.sum(y == T) >= np.sum(yp == T) else -1

for i in range(maxiter):

    converged = True
    if i == 0:
        w[0] = np.array(w[i] + alpha * T[i] * X[i][0])
        for k in range(N-1):
            w = np.append(w, w[k] + alpha * T[k] * X[k][0])
    else:
        w[0] += alpha * T[0] * X[0][0]
        for k in range(N-1):
            w[k+1] = w[k] + alpha * T[k] * X[k][0]

    # Converged?
    error = (1/len(T)) * np.sum([np.sign(w[i] * X[i]) != T[i] for i in range(len(T))])
    if error != 0:
        converged = False
        if last_lowest_error > error:
            print('Setting last_lowest to {error}')
            last_lowest_error = error
            w_pocket = copy(w)

    if converged:
        print("converged at ", i)
        break

w = copy(w_pocket)
print("End of training: ", i)
plt.plot(T)
plt.plot([_[0] for _ in X] * w)
plt.show()

# show decision boundary
plt.plot(C1[:, 0], C1[:, 1], 'or')
plt.plot(C2[:, 0], C2[:, 1], 'xb')


xt = np.array([-2, 5])
yt = -w[0] * xt / w[-1]

plt.plot(xt, yt)
plt.xlim([-3, 6])
plt.ylim([-3, 7])
plt.show()