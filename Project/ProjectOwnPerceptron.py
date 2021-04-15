# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:37:29 2021

@author: Spencer Peterson
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import os
import sys

PerceptronPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Perceptron'))
sys.path.append(PerceptronPath)
from Perceptron import Perceptron

train = 'train_final.csv'
test = 'test_final.csv'


df = pd.read_csv(train)


def ProcessTrainingDataFrame(df):
         # Check if each column is all numeric
    
    df = df.astype(object)
    
    df_yes = df[df[df.columns[14]]==1]
    
    for colInd in range(0,len(df.columns)-1):
        currMode = df_yes[df_yes.columns[colInd]].mode()[0]
        df_yes[df_yes.columns[colInd]].replace({'?' : currMode}, inplace=True)
    
    
    df_no = df[df[df.columns[14]]==0]
        # Change unknowns to most common value in column                
     
    for colInd in range(0,len(df.columns)-1):
        currMode = df_no[df_no.columns[colInd]].mode()[0]
        df_no[df_no.columns[colInd]].replace({'?' : currMode}, inplace=True)
        
        
    return pd.concat((df_yes, df_no))


def ProcessTestingDataFrame(df):
    df = df.astype(object)


    for colInd in range(0,len(df.columns)):
        currMode = df[df.columns[colInd]].mode()[0]
        df[df.columns[colInd]].replace({'?' : currMode}, inplace=True)
    
    
    return df


df = ProcessTrainingDataFrame(df)

#df = df.drop(['native.country', 'education'], axis=1)


labelProcessor = preprocessing.LabelEncoder()

for i in range(len(df.columns)):
    df.iloc[:,i] = labelProcessor.fit_transform(df.iloc[:,i])
# %% 

from AveragedPerceptron import AveragedPerceptron

averager = AveragedPerceptron(epochs = 10, r = 0.5)

averager.BuildPerceptron(df)

# %% 
num_wrong = 0
num_total = df.shape[0]
for i in range(int(df.shape[0]/2)):
    pred = averager.Predict(df.iloc[i,:])
    
    if df.iloc[i,-1] != pred:
        num_wrong += 1
        
print('Training error was:', num_wrong/num_total)
# %%

testDF = pd.read_csv(test)
testDF = ProcessTestingDataFrame(testDF)

for i in range(len(testDF.columns)):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])
testDF['fake_results'] = pd.Series(0 for x in range(len(testDF.index)))
# %% 
predictionDF = pd.DataFrame(columns = ["ID", "Prediction"])


for i in range(int(testDF.shape[0])):
    pred = averager.Predict(testDF.iloc[i,1:])

    predictionDF.loc[len(predictionDF.index)] = [testDF["ID"][i]+1, pred]
        
predictionDF.to_csv('Predictions_OwnPerceptron.csv', index=False)

# %% 

Y = df.iloc[:,-1]
X = df.iloc[:,0:14]

Y = Y.astype(int)
mlp = MLPClassifier(random_state=1, hidden_layer_sizes=[20]*7)
mlp.fit(X,Y)

test_predictions = mlp.predict(X)
print(accuracy_score(Y, test_predictions))

testDF = pd.read_csv(test)
testDF = ProcessTestingDataFrame(testDF)

for i in range(1,15):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])



predictions = mlp.predict(testDF.iloc[:,1:15])

predictionDF = pd.DataFrame(predictions)

predictionDF["ID"] = testDF["ID"].values
predictionDF.to_csv('Predictions_MLP.csv', index=False, header=['Prediction','ID'])