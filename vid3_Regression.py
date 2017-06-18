import pandas as pd
# quandl is an api for querying a companies
# database for finance information
import math, quandl
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

# returns an array of the stocks of google
df = quandl.get('WIKI/GOOGL')

# This displays all of the stock data
# print(df.head())

# Since the df is an array, and since we do not care about
# the columns such as Split ratio, Ex-Dividend, ect..
# we create a new version of the array, but with only the
# rows that we want
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# This finds the high low percent of each row, or daily percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

# The amount that the close and open price changes
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

forecast_col = 'Adj. Close'
# in machine learning you do not want to use nan data
df.fillna(-99999, inplace=True)

# rounds everything up to a nice whole number
forecast_out = int(math.ceil(0.01 * len(df)))

# shifts the columns to be 10 days into the future, or 10%
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

# features will be a capital X, and y is the labels
X = np.array(df.drop(['label'], 1))
y = np.array(['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# you can set the number of threads to run also
# such as clf = LinearRegression(n_jobs=10) or -1 to run as many as possible
# some can be threaded which can be useful
clf = LinearRegression()
# clf = svm.SVR(kernel='poly') you can switch algorithms easily
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
