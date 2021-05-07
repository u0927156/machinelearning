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
from sklearn.model_selection import train_test_split

# %% Define projection function

def ExpandDataframe(df):
    len_cols = len(df.columns)
    
    for i in range(len_cols):
        for j in range(1, len_cols):
            df[len(df.columns)] = df.iloc[:,i] * df.iloc[:,j]
            

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

# %% Load and preprocess data
train = 'train_final.csv'
test = 'test_final.csv'


df = pd.read_csv(train)

df = ProcessTrainingDataFrame(df)

labelProcessor = preprocessing.LabelEncoder()

for i in range(14):
    df.iloc[:,i] = labelProcessor.fit_transform(df.iloc[:,i])



Y = df.iloc[:,-1]
X = df.iloc[:,0:14]

ExpandDataframe(X)



# %% Cross Validation

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

depths = [3, 7, 11, 17, 23, 29]
ns = [5, 10, 50, 75, 100, 300]

depth=depths[0]
n = ns[0]

for depth in depths:
    for n in ns:
        decForest = RandomForestClassifier(max_depth=depth, n_estimators=n)
        decForest = decForest.fit(X_train,Y_train)
        
        training_predictions = decForest.predict(X_train)
        train_acc = accuracy_score(Y_train, training_predictions)
        
        
        testing_predictions = decForest.predict(X_test)
        test_acc = accuracy_score(Y_test, testing_predictions)
        
        print('%d & %d & %.4f & %.4f \\\\ \hline' % (depth, n, train_acc, test_acc))
# %% Train model with chosen parameters


Y = Y.astype(int)
decForest = RandomForestClassifier(max_depth=11, n_estimators=300)
decForest = decForest.fit(X,Y)


test_predictions = decForest.predict(X)
print(accuracy_score(Y, test_predictions))

# %% test
testDF = pd.read_csv(test)

testDF = ProcessTestingDataFrame(testDF)


for i in range(1,15):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])
    
X_test = testDF.iloc[:,1:15]
ExpandDataframe(X_test )

predictions = decForest.predict(X_test)

predictionDF = pd.DataFrame(predictions)
predictionDF["ID"] = testDF["ID"].values
predictionDF.to_csv('Predictions_2.csv', index=False, header=['Prediction','ID'])



    
# %%



prob_predictions = decForest.predict_proba(X_test)


predictionDF = pd.DataFrame(prob_predictions[:,1])

predictionDF["ID"] = testDF["ID"].values

predictionDF.to_csv('PredictionsForestProbability.csv', index=False, header=['Prediction','ID'])
