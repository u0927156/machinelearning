# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:40:17 2021

@author: Spencer Peterson
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

train = 'train_final.csv'
test = 'test_final.csv'


df = pd.read_csv(train)

Y = df.iloc[:,-1]
X = df.iloc[:,0:14]

labelProcessor = preprocessing.LabelEncoder()

for i in range(14):
    df.iloc[:,i] = labelProcessor.fit_transform(df.iloc[:,i])



Y = df.iloc[:,-1]
X = df.iloc[:,0:14]

decForest = RandomForestClassifier(max_depth=6, n_estimators=150)
decForest = decForest.fit(X,Y)

testDF = pd.read_csv(test)


test_predictions = decForest.predict(X)
print(accuracy_score(Y, test_predictions))


for i in range(1,15):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])
    

predictions = decForest.predict(testDF.iloc[:,1:15])

predictionDF = pd.DataFrame(predictions)
predictionDF["ID"] = testDF["ID"].values
predictionDF.to_csv('Predictions_2.csv', index=False, header=['Prediction','ID'])
