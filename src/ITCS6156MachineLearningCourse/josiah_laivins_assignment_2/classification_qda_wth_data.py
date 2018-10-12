# from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.QDAClassifier import QDAClassifier
# from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.util.display_util import *
# import numpy as np
# import matplotlib.pyplot as plt
#
# mu1 = [-1, -1]
# cov1 = np.eye(2)
#
# mu2 = [2,3]
# cov2 = np.eye(2) * 3
#
# C1 = np.random.multivariate_normal(mu1, cov1, 50)
# C2 = np.random.multivariate_normal(mu2, cov2, 50)
#
# plt.plot(C1[:, 0], C1[:, 1], 'or')
# plt.plot(C2[:, 0], C2[:, 1], 'xb')
#
# plt.xlim([-3, 6])
# plt.ylim([-3, 7])
# plt.show()
#
# big_x = np.vstack((C1, C2))
# big_t = np.ones(100)
# big_t[:50] *= -1
#
# clf = QDAClassifier()
# clf.train(big_x, big_t)
#
# show_boundaries(clf.discriminant_functions[-1]-
#                              clf.discriminant_functions[1])
#
# test_big_x = [[0, 0], [5, 5], [2, 5], [8,8]]
# print(f'{clf.use(test_big_x)}')
# show_accuracy(clf.use(big_x), big_t)

from pathlib import Path
import os
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.QDAClassifier import QDAClassifier
from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.util.display_util import *
import pandas as pd
import numpy as np

# Load data:
base_dir = str(Path().absolute())
n_rows = 100  # None for all
data = pd.read_csv(base_dir + os.sep + 'data' + os.sep + 'stack-overflow-2018-developer-survey' + os.sep +
                   'survey_results_public.csv', nrows=n_rows)

# Features of interest:
features = ['SkipMeals', 'WakeTime', 'HoursComputer', 'RaceEthnicity', 'CareerSatisfaction']  # Removed: 'JobSatisfaction' because... that's too easy
# features = ['SkipMeals', 'WakeTime', 'CareerSatisfaction']

# Filter Features:
data = data[features]

# We want to predict CareerSatisfaction
classification = 'CareerSatisfaction'
# We want this classification to be binary. We will range it from not satisfied to satisfied -1 to +1
replacement_keys = {'Extremely satisfied': 1, 'Neither satisfied nor dissatisfied': 1,
                    'Moderately satisfied': 1, 'Slightly dissatisfied': -1, 'Slightly satisfied': 1,
                    'Moderately dissatisfied': -1, 'Extremely dissatisfied': -1}

data = data.replace({classification: replacement_keys})  # Target is now binary

data = data.dropna(axis=0).reset_index(drop=True)  # Drop Null or nan records.
print(f'Rows was {n_rows} before dropping, but now is: {data.shape[0]}')
# Convert String columns into one-hot encoded columns
for column in data:
    if data[column].dtype == object:
        # print(f'Encoding one-hot for column: {column} \r')
        data[column] = LabelEncoder().fit_transform(y=data[column].fillna('0'))  # TODO save an array of LabelEncoders

# Show correlation Matrix for this data set
show_correlation(features, data, n_rows, is_labeled=True)

# Split the data into features and targets
x = pd.DataFrame.copy(data.drop(classification, axis=1))  # Exclude the classification field from the training samples
y = pd.DataFrame.copy(data.drop([f for f in features if f != classification], axis=1))
print("Split data")

# Split the features into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.33)

clf = QDAClassifier()
clf.train(X_train, y_train)

print(f'{clf.use(X_test)}')
