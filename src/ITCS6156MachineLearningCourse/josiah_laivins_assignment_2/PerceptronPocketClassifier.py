from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.BaseModel import Classifier
import numpy as np


class PerceptronPocketClassifier(Classifier):

    def __init__(self, max_iterations: int, alpha: float = 0.1) -> object:
        """

        :param max_iterations:
        :param alpha:
        """
        super().__init__()
        self.alpha = alpha  # Learning rate
        self.max_iterations = max_iterations  # training iterations
        # Weight matrix
        self.w = np.random.uniform(-1.0, 1.0, 1).reshape(-1, 1)
        self.w_pocket = np.copy(self.w)

    def _compare(self, x, targets):
        y = np.sign(x @ self.w)
        yp = np.sign(x @ self.w_pocket)

        return 1 if np.sum(y == targets) >= np.sum(yp == targets) else -1

    def train(self, x: np.ndarray, targets: np.ndarray):
        # Set Shape positions
        shape_features = x.shape[1]
        shape_num_samples = x.shape[0]
        shape_target_features = targets.shape[1]

        # Reset the w to reflect the dimensions being trained on
        self.w = np.zeros(shape_features).reshape(-1, 1)#np.random.uniform(-1.0, 1.0, shape_features).reshape(-1, shape_target_features)
        self.w_pocket = np.copy(self.w)

        # Normalize Training Data:
        # x = self.normalize(x, reset_fields=True)

        for j in range(self.max_iterations):
            print(f'Iteration: {j}')
            converged = True
            for k in np.random.permutation(shape_num_samples - 1):
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
                y = np.transpose(self.w) @ x[k]

                if np.sign(y) != np.sign(targets[k]):
                    self.w += self.alpha * x[k].reshape(-1, 1) * targets[k].reshape(-1, 1)
                    converged = False
                    if self._compare(x, targets) > 0:
                        self.w_pocket[:] = self.w[:]

            if converged:
                print("converged at ", j)
                break

    def use(self, x: np.ndarray):
        # x = self.normalize(x)
        return x @ self.w_pocket
