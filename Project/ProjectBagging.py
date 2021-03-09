# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:51:54 2021

@author: Spencer Peterson
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
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

bagger = BaggingClassifier(n_estimators=300, bootstrap_features=True)
bagger = bagger.fit(X,Y)

testDF = pd.read_csv(test)


test_predictions = bagger.predict(X)
print(accuracy_score(Y, test_predictions))


for i in range(1,15):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])
    

predictions = bagger.predict(testDF.iloc[:,1:15])

predictionDF = pd.DataFrame(predictions)
predictionDF["ID"] = testDF["ID"].values
predictionDF.to_csv('Predictions_4.csv', index=False, header=['Prediction','ID'])
