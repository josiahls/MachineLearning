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
"""
import numpy as np
from sklearn import preprocessing, model_selection, neighbors

import pandas as pd


accuracies = []

for i in range(5):
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
    X = np.array(df.drop(['class'],1))
    # labels
    y = np.array(df['class'])

    # sets values from model generation
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test,y_test)
    #print(accuracy)

    # example data to test against. This will be 2
    #example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
    #example_measures = example_measures.reshape(len(example_measures),-1)

    #prediction = clf.predict(example_measures)
    #print(prediction)
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))