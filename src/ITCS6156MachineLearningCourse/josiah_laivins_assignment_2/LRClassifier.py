import numpy as np
from sklearn.preprocessing import LabelBinarizer

from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.BaseModel import Classifier


# noinspection PyMethodMayBeStatic
class LRClassifier(Classifier):
    def __init__(self):
        super().__init__()

        self.w = None
        self.binarizer = LabelBinarizer()

    def _soft_max(self, z: np.ndarray):
        if not isinstance(z, np.ndarray):
            z = np.asarray(z)
        f = np.exp(z)
        return f / (np.sum(f, axis=1, keepdims=True)) if len(z.shape) == 2 else np.sum(f)

    def _g(self, big_x):
        return self._soft_max(big_x @ self.w)

    def train(self, big_x: np.ndarray, big_t: np.ndarray, iterations=1000, alpha=0.1):
        big_x = (big_x - np.mean(big_x, 0)) / np.std(big_x, 0)

        # Fix big_t to be one hot (one column for each class)
        one_hot_big_t = self.one_hot(big_t)

        n_samples = big_x.shape[0]
        n_features = big_x.shape[1]
        n_target_dims = one_hot_big_t.shape[1]
        n_bias_dims = 1
        self.w = np.random.rand(n_features + n_bias_dims, n_target_dims)

        # Add a bias column to big_x
        bias_big_x = np.hstack((np.ones((n_samples, n_bias_dims)), big_x))

        for step in range(iterations):
            y_scaled = self._g(bias_big_x)
            self.w += alpha * bias_big_x.T @ (one_hot_big_t - y_scaled)

    def one_hot(self, big_t: np.ndarray):
        # One hot via numpy:
        self.binarizer.fit(big_t)
        labels = self.binarizer.transform(big_t)
        return np.hstack((labels, 1 - labels))

    def use(self, big_x: np.ndarray):
        big_x = (big_x - np.mean(big_x, 0)) / np.std(big_x, 0)
        n_samples = big_x.shape[0]
        n_bias_dims = 1
        bias_big_x = np.hstack((np.ones((n_samples, n_bias_dims)), big_x))

        y = self._g(bias_big_x)

        return self.binarizer.inverse_transform(np.argmax(y, axis=1))
