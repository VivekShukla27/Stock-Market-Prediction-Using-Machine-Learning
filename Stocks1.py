# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:30:30 2019

@author: MADY
"""

import pandas as pd
import numpy as np
data= pd.read_csv("stock.csv")
print(data.head)
data= data.as_matrix()
X= data[:,[2]]
y= data[:,-1]
y=y.astype('int')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)
print(X_train)
print(y_train)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, y_train)
p= lr.predict(X_test)
print("p=",p)


from sklearn.tree import DecisionTreeClassifier
clf_entropy = DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5) 
clf_entropy.fit(X_train, y_train) 
e= clf_entropy.predict(X_test)
print("e=",e)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

from sklearn import metrics 
print(metrics.mean_absolute_error(y_test,p))
print(metrics.mean_squared_error(y_test,p))
print(np.sqrt(metrics.mean_squared_error(y_test,p)))

print("p=",p)
print("e=",e)
print("pred=",pred)

import matplotlib.pyplot as plt
plt.figure(2)

plt.subplot(221)
plt.title("stock market")
plt.plot(y_test,p)

plt.subplot(222)
plt.plot(y_test,e)

plt.subplot(223)
plt.plot(y_test,pred)


plt.subplot(224)
plt.plot(y_test,pred)

plt.show()
