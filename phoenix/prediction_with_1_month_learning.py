import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pandas as pd
import math, datetime
from matplotlib import style
import re
from datetime import datetime
import time

style.use('ggplot')

dt = pd.read_csv("Project-3.csv")

dt['Time'] = pd.to_datetime(dt['Time'], format='%d/%m/%y %H:%M')

dt.index = dt['Time']
del dt['Time']
dt1 = dt['2017-07':'2017-08']

X = np.array(dt1['x'])
y = np.array(dt1['y'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)

clf = LinearRegression(n_jobs=-1)
X_train = X_train.reshape(-1,1)
clf.fit(X_train, y_train)

X_test = X_test.reshape(-1,1)
accuracy = clf.score(X_test, y_test)
print(accuracy)

X = np.array(dt['x'])
X = X.reshape(-1,1)
y = np.array(dt['y'])
t = clf.predict(X)
print(t)

plt.plot(y,'r')
plt.plot(t,'b')
plt.show()

