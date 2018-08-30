"""
K-Means
Has one param: K which is the max number of groups

K=2 is flat clustering

Takes a data set, randomly sets centroids
Calculates the distance of each feature to the centroids
and classify each feature set based on which centroid they are closest to
then take the mean of the distances
set the centroids to the new location (mean)

optimized is when a centroid barely moves

negative is it tends to try to have equally sized groups
"""

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()

clf = KMeans(n_clusters=3)

clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = 10*["g.", "r.", "c.", "b.", "k."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)

plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
plt.show()
