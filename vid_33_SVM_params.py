"""
OVR = one verse rest
OVO = one versus one

These are 2 ways of separating groups

OVR means separating one group from everything else (other data, and groups)
OVO used more: creates multiple vs multiple groups per group. Can be more useful
               but needs more processing. 


"""
import numpy as np
from sklearn import preprocessing, neighbors, svm, model_selection
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
