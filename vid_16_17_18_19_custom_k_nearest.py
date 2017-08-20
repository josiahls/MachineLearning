"""
SEE:
   This breast cancer databases was obtained from the University of Wisconsin
   Hospitals, Madison from Dr. William H. Wolberg.  If you publish results
   when using this database, then please include this information in your
   acknowledgements.  Also, please cite one or more of:

   1. O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear
      programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18.

   2. William H. Wolberg and O.L. Mangasarian: "Multisurface method of
      pattern separation for medical diagnosis applied to breast cytology",
      Proceedings of the National Academy of Sciences, U.S.A., Volume 87,
      December 1990, pp 9193-9196.

   3. O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition
      via linear programming: Theory and application to medical diagnosis",
      in: "Large-scale numerical optimization", Thomas F. Coleman and Yuying
      Li, editors, SIAM Publications, Philadelphia 1990, pp 22-30.

   4. K. P. Bennett & O. L. Mangasarian: "Robust linear programming
      discrimination of two linearly inseparable sets", Optimization Methods
      and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers).

This script uses the above data to test k-nearest neighbors

It uses euclidean distance willbe used:

Is: sqrt ( sum n ((qi - Pi)^2) i = 1 )
where q is the first point 1,3
P is 2,5

So: sqrt ( (1 - 2)^2 + (3 - 5)^2  )
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd
import random


# style.use('fivethirtyeight')

# dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
# new_features = [5, 7]

# or [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# for i in dataset:
#   for ii in dataset[i]:
#      plt.scatter(ii[0], ii[1], s=100, color=i)

# plt.scatter(new_features[0],new_features[1])
# plt.show()

# euclid_distance = sqrt((plot1[0] - plot2[0])**2 + (plot1[1] -plot2[1])**2)

# print(euclid_distance)

# data is thedata weare running k nearest on
# predict is the features we want to group
# k is the number of comparison we want to make with the dataset
def k_nearest_neighbors(data, predict, k=3):
    # checks if the dataset is stupidly low
    # like 2 sets of data when k is 3
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups idiot')

    distances = []
    for group in data:
        for features in data[group]:
            # alt: euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict)) ** 2))
            # creates a list of distances and groups
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    # for i in sorted(distances)[:k]:
    #   i[1]

    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    #print(vote_result, confidence)

    return vote_result, confidence


"""
result= k_nearest_neighbors(dataset, new_features, k=3)
print(result)

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)

plt.scatter(new_features[0],new_features[1], color=result)
plt.show()
"""
accuracies = []

for i in range(5):
    df = pd.read_csv("breast-cancer-wisconsin.data.txt")
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    # convert to a float (makes sure all values are numeric)
    full_data = df.astype(float).values.tolist()

    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    # first 80% of data
    train_data = full_data[:-int(test_size * len(full_data))]
    # last 20% of data
    test_data = full_data[-int(test_size * len(full_data)):]

    for i in train_data:
        # appending lists into this list
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        # appending lists into this list
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            #else:
              #  print(confidence)
            total += 1

    print('Accuracy:', correct / total)
    accuracies.append(correct/total)

print(sum(accuracies) / len(accuracies))