from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.LDAClassifier import LDAClassifier
from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.util import display_util
import numpy as np
import matplotlib.pyplot as plt

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

clf = LDAClassifier()
clf.train(big_x, big_t)

display_util.show_boundaries(clf.discriminant_functions[-1]-
                             clf.discriminant_functions[1])

test_big_x = [[0, 0], [5, 5], [2, 5], [8,8]]
print(f'{clf.use(test_big_x)}')
