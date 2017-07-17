import pandas as pd
import quandl , math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100


df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)
print(df.head())


x = np.array(df.drop(['label'],1))
Y = np.array(df['label'])
             
x = preprocessing.scale(x)

x = x[:-forecast_out+1]
df.dropna(inplace = True)
Y = np.array(df['label'])
X_train, X_test , Y_train, Y_test = cross_validatoin.train_testsplit(x,Y,test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,Y_train)
accuracy = clf.score(X_test,Y_test)

print(accuracy)
    
