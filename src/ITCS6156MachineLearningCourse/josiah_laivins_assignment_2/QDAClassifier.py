from typing import List, Any

from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.BaseModel import Classifier
import numpy as np


# noinspection PyMethodMayBeStatic
class QDAClassifier(Classifier):
    def __init__(self):
        super().__init__()

        self.discriminant_functions = {}
        self.discriminant_function_params = {}

        self.global_mean = 0
        self.global_stds = 0

    def train(self, big_x: np.array, big_t: np.array):
        # Scale the big_x sample base
        self.global_mean, self.global_stds = np.mean(big_x, 0), np.std(big_x, 0)
        scaled_big_x = (big_x - self.global_mean) / self.global_stds

        # Split them by their classes
        for unique in set(big_t):
            # Get the samples that have that unique value
            indexes = np.where(big_t == unique)
            temp_x = np.copy(scaled_big_x[indexes])
            mu = np.mean(temp_x, 0)
            sigma = np.cov(temp_x.T)
            prior = float((len(big_t[big_t == unique]) / len(big_t)))
            # Get and save the discriminant function

            xs, ys = np.meshgrid(np.linspace(-3, 6, 500), np.linspace(-3, 7, 500))
            Xtest = np.vstack((xs.flat, ys.flat)).T
            XtestS = (Xtest - self.global_mean) / self.global_stds
            self.discriminant_functions[unique] = self.get_qda(XtestS, mu, sigma, prior)
            self.discriminant_function_params[unique] = (mu, sigma, prior)

    def get_qda(self, big_x: np.array, mu, sigma, prior):
        sigma_inv = np.linalg.inv(sigma)
        diff_v = big_x - mu
        return - 0.5 * np.log(np.linalg.det(sigma)) \
               - 0.5 * np.sum(diff_v @ sigma_inv * diff_v, axis=1) + np.log(prior)

    def use(self, big_x):

        scaled_big_x = (big_x - self.global_mean) / self.global_stds
        classes = [c for c in self.discriminant_function_params]
        evaluations = []
        for sample in scaled_big_x:
            probabilities: List[float] = []
            for class_value in self.discriminant_function_params:
                print(f'\n\nSample: {sample} Class to test: {class_value} the resulting prob: '
                      f'{self.get_qda(np.array(sample).reshape(-1, 1), *self.discriminant_function_params[class_value])}')

                probabilities\
                    .append(max(self.get_qda(np.array(sample).reshape(-1, 1),
                                             *self.discriminant_function_params[class_value])))
            evaluations.append(classes[np.argmax(probabilities)])
        return evaluations
