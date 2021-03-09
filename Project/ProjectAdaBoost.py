# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:39:09 2021

@author: Spencer Peterson
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree

train = 'train_final.csv'
test = 'test_final.csv'


df = pd.read_csv(train)

Y = df.iloc[:,-1]
X = df.iloc[:,0:14]

labelProcessor = preprocessing.LabelEncoder()

for i in range(14):
    df.iloc[:,i] = labelProcessor.fit_transform(df.iloc[:,i])



y = df.iloc[:,-1]
X = df.iloc[:,0:14]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

AdaBooster = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(max_depth=2))
AdaBooster = AdaBooster.fit(X,y)

testDF = pd.read_csv(test)


test_predictions = AdaBooster.predict(X_test)
print(accuracy_score(y_test, test_predictions))


for i in range(1,15):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])
    

predictions = AdaBooster.predict(testDF.iloc[:,1:15])

predictionDF = pd.DataFrame(predictions)
predictionDF["ID"] = testDF["ID"].values
predictionDF.to_csv('Predictions_3.csv', index=False, header=['Prediction','ID'])
