"""
We will be working in vectors.
Is better than neural networks at
certain tasks.

Binary clissifier that tries to separate into 2 groups:
positive and negative. It tries to find the best separation
boundry(best separating hyper plane). It is basically trying ti
find a line that is farthest from each element but between eachgroup.

This can make classification easy since unknown elements
can just be compared to the hyperplane.

"""
import numpy as np
from sklearn import preprocessing, model_selection, svm

import pandas as pd

# import text file as a csv
# converts into array
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# convert '?' into numerical data
df.replace('?', -999999, inplace=True)
# remove the id column (note 1 means that we are)
# dropping the column for each row. In theory you
# could have 0 instead. Same behavior unless
# there are differing column lengths
df.drop(['id'], 1, inplace=True)

# features
X = np.array(df.drop(['class'], 1))
# labels
y = np.array(df['class'])

# sets values from model generation
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2,3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)
