import numpy as np

def normald(X, mu, sigma):
    """
    Calculates the normal distributions

    :param X: N x D, D features, N samples
    :param mu: mean is D features x 1 because it is a single sample that is the mean of all of the o
    other samples.
    :param sigma: Converience matrix. Not sure what this is, but it is D x D for features
    :return:
    """

    D_features = X.shape[1]

    det_sigma = sigma if D_features == 1 else np.linalg.det(sigma)

    if det_sigma == 0:
        raise np.linalg.LinAlgError('normald(): Singular matrix')

    sigmaI = 1.0 / sigma if D_features == 1 else np.linalg.inv(sigma)

    norm_constant = 1.0 / np.sqrt((2*np.pi) ** D_features * det_sigma)

    diffv = X - mu.T

    return norm_constant * np.exp(-0.5 * np.sum(np.dot(diffv, sigmaI) * diffv, axis=1))[:,np.newaxis]

def calculate_convariance_matrix(X, mu):
    # https://en.wikipedia.org/wiki/Covariance_matrix
    # fixed broadcasting issue via: https://stackoverflow.com/questions/34447714/simple-subtraction-causes-a-broadcasting-issue-for-different-array-shapes
    adjusted_mu = mu[:, :]

    return np.cov(X, rowvar=False)


X = np.array([[1, 2], [3,5 ], [2.1, 1.9]])
mu = np.array([[2], [2]])
Sigma = np.array([[1, 0], [0, 1]])
# Sigma = calculate_convariance_matrix(X, mu)

print(X)
print(mu)
print(Sigma)
print(normald(X, mu, Sigma))