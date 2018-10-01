import numpy as np
import matplotlib.pyplot as plt

# Class features
X = np.array([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4]
])

# Target classifications
# # Reshape changes 8, to 8, 1
T = np.array([5, 6, 5, 6, 5, 6, 5, 6]).reshape(-1, 1)
T[T == 6] = 1
T[T == 5] = -1



mu1 = [-1, -1]
cov1 = np.eye(2)

mu2 = [2,3]
cov2 = np.eye(2) * 3

C1 = np.random.multivariate_normal(mu1, cov1, 50)
C2 = np.random.multivariate_normal(mu2, cov2, 50)

X = np.vstack((C1, C2))
N = X.shape[0]
T = np.ones(N).reshape(-1, 1)
T[:50] *= -1



# Set Shape positions
shape_features = X.shape[1]
shape_num_samples = X.shape[0]
shape_target_features = T.shape[1]

# Max iterations
max_iter = 8000
# Learning rate
alpha = 0.1

# Weight matrix
W = np.random.uniform(-1.0, 1.0, shape_features).reshape(-1, shape_target_features)
w_pocket = np.copy(W)

def compare(X, T, w, wp):
    y = np.sign(X @ w)
    yp = np.sign(X @ wp)

    return 1 if np.sum(y == T) >= np.sum(yp == T) else -1

def train(X: np.array, T: np.array, W: np.array):

    for j in range(max_iter):
        converged = True
        for k in range(shape_num_samples - 1):
            '''
            ok so, we need weight (W) to be (target_dim X feature_num)
            so....
            T[k] * X[k] is not enough. Note that this would output (feature X 1)
            This is too constraining. This means that each target has to be a scalar,
            and X cant have upper dimensionality. This also does not take advantage of matrix 
            operations.
            
            So we fix this by:
            
            transpose(T dot transpose(X))
            
            because T is (t_features X constant) and X is (features X constant) 
            and we want (features X constant) outputted
            
            '''
            y = np.transpose(W) @ X[k]

            if np.sign(y) != np.sign(T[k]):
                W += alpha * X[k].reshape(-1, 1) * T[k].reshape(-1, 1)

                if compare(X, T, W, w_pocket) > 0:
                    w_pocket[:] = W[:]

                if converged:
                    print("converged at ", j)
                    break


train(X, T, W)

plt.plot(T, label="Ground Truth")
plt.plot(X@w_pocket, label="Result")
plt.legend()
plt.show()

print()
