# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:58:05 2021

@author: Spencer Peterson
"""


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
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

CNN = MLPClassifier(alpha=10e-6, random_state=0)
CNN = CNN.fit(X,Y)

testDF = pd.read_csv(test)


test_predictions = CNN.predict(X)
print(accuracy_score(Y, test_predictions))


for i in range(1,15):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])
    

predictions = CNN.predict(testDF.iloc[:,1:15])

predictionDF = pd.DataFrame(predictions)
predictionDF["ID"] = testDF["ID"].values
predictionDF.to_csv('Predictions_5.csv', index=False, header=['Prediction','ID'])
