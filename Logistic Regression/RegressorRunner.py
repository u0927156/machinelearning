# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:29:53 2021

@author: Spencer Peterson
"""
import os
import sys
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt


PerceptronPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Perceptron'))
LogisticPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Logistic Regression'))
sys.path.append(PerceptronPath)
sys.path.append(LogisticPath)

from MAPRegressor import MAPRegressor

train = os.path.abspath(os.path.join(PerceptronPath, 'train.csv')) # located in the peceptron file
test = os.path.abspath(os.path.join(PerceptronPath, 'test.csv'))

# %%
from MAPRegressor import MAPRegressor
df = pd.read_csv(train, header=None)

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

for var in variances: 
    ImTheMap = MAPRegressor(var = var, gamma_0 = .5, d = 1)
    
    ImTheMap.buildRegressor(df)
    
    
    
    num_wrong = 0
    num_total = df.shape[0]
    for i in range(df.shape[0]):
        pred = ImTheMap.Predict(df.iloc[i,:])
        
        if df.iloc[i,-1] != pred:
            num_wrong += 1
            
    print('For variance =', var)
    print('Training error was:', num_wrong/num_total)
    
    test_df = pd.read_csv(test, header=None)
    
    num_wrong = 0
    num_total = test_df.shape[0]
    for i in range(test_df.shape[0]):
        pred = ImTheMap.Predict(test_df.iloc[i,:])
        
        #print(pred, test_df.iloc[i,-1])
        if test_df.iloc[i,-1] != pred:
            num_wrong += 1
            
    print('Testing error was:', num_wrong/num_total)
    
    


fig, ax = plt.subplots()
ax.plot(ImTheMap.num_updates, ImTheMap.ObjectiveFunction)
plt.show()


# %%
from MLERegressor import MLERegressor
df = pd.read_csv(train, header=None)


MLE = MLERegressor(var = .01, gamma_0 = 0.1, d = 10)

MLE.buildRegressor(df)



num_wrong = 0
num_total = df.shape[0]
for i in range(df.shape[0]):
    pred = MLE.Predict(df.iloc[i,:])
    
    if df.iloc[i,-1] != pred:
        num_wrong += 1
        
print('Training error was:', num_wrong/num_total)

test_df = pd.read_csv(test, header=None)

num_wrong = 0
num_total = test_df.shape[0]
for i in range(test_df.shape[0]):
    pred = MLE.Predict(test_df.iloc[i,:])
    
    #print(pred, test_df.iloc[i,-1])
    if test_df.iloc[i,-1] != pred:
        num_wrong += 1
        
print('Testing error was:', num_wrong/num_total)


