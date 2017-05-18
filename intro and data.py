import pandas as pd
# quandl is an api for querying a companies
# database for finance information
import quandl

# returns an array of the stocks of google
df = quandl.get('WIKI/GOOGL')

# This displays all of the stock data
print(df.head())

# Since the df is an array, and since we do not care about
# the columns such as Split ratio, Ex-Dividend, ect..
# we create a new version of the array, but with only the
# rows that we want
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# This finds the high low percent of each row, or daily percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

# The amount that the close and open price changes
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

print('\n\n Cleaned Dataset\n\n')

# Creates the new cleaner array
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# dispay the new dataset
print(df.head())
