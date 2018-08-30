import pandas as pd
import math, quandl, datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

#use to display data
style.use('ggplot')





df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

print(df.head())

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
# the stuff we will predict against
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)
y = np.array(['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

"""NEW CODE"""
# this is a big important thing is sykit learn
forecast_set = clf.predict(X_lately)
# These output the next 30 days of prices
print(forecast_set, accuracy)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# this is here so you can have dates on the axis
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # the date is the index of that data frame
    # anything that is not a number () the labels
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


