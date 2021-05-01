# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:10:24 2021

@author: Spencer Peterson
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy

SVMPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'SVM'))
sys.path.append(SVMPath)
from PrimalSVM import PrimalSVM

train = os.path.abspath(os.path.join(SVMPath, 'train.csv')) # located in the peceptron file
test = os.path.abspath(os.path.join(SVMPath, 'test.csv'))


df = pd.read_csv(train, header=None)

test_df = pd.read_csv(test, header=None)

# %% gamma_t = gamma_0 / (1 + (gamma_0/d) * t)
print('gamma_t = gamma_0 / (1 + (gamma_0/d) * t)')
C = 700/873;
SVM = PrimalSVM(epochs = 100,learning_schedule=0, C=C)

SVM.BuildSVM(df)

num_wrong = 0
num_total = df.shape[0]
for i in range(df.shape[0]):
    pred = SVM.Predict(df.iloc[i,:])
    
    if df.iloc[i,-1] != pred:
        num_wrong += 1
        

print('Training error was:', num_wrong/num_total)


test_df = pd.read_csv(test, header=None)

num_wrong = 0
num_total = test_df.shape[0]
for i in range(test_df.shape[0]):
    pred = SVM.Predict(test_df.iloc[i,:])
    
    if test_df.iloc[i,-1] != pred:
        num_wrong += 1
        

print('Testing error was:', num_wrong/num_total)

print('The weight vector was: $', SVM.w, '$.\n')

# gammma = gamma_0 / (1+ t)

print('gamma / (1+t)')
SVM = PrimalSVM(epochs = 100,learning_schedule=1, C=C)

SVM.BuildSVM(df)

num_wrong = 0
num_total = df.shape[0]
for i in range(df.shape[0]):
    pred = SVM.Predict(df.iloc[i,:])
    
    if df.iloc[i,-1] != pred:
        num_wrong += 1
        

print('Training error was:', num_wrong/num_total)


test_df = pd.read_csv(test, header=None)

num_wrong = 0
num_total = test_df.shape[0]
for i in range(test_df.shape[0]):
    pred = SVM.Predict(test_df.iloc[i,:])
    
    if test_df.iloc[i,-1] != pred:
        num_wrong += 1
        

print('Testing error was:', num_wrong/num_total)

print('The weight vector was: $', SVM.w, '$.')

# %% Dual SVM

gamma_list = [.1, .5, 1, 5, 100]
for gamma in gamma_list:
    
    def GaussianKernel(x1, x2):
        if x1.ndim>1 or x2.ndim > 1:
            return np.exp(-np.power(np.linalg.norm(x1 - x2, axis=1), 2) / gamma)
        else:
            return np.exp(-np.power(np.linalg.norm(x1 - x2), 2) / gamma)
    
    from DualSVM import DualSVM
    
    
    
    print('Gamma =', gamma)
        
    for C in [100, 500, 700]:
        
        
        
        SVM = DualSVM(C = C/873, kernel = GaussianKernel)
        
        #num_examples = 200;
        SVM.BuildSVM(df.iloc[:,:])
        
        """
        num_wrong = 0
        num_total = df.shape[0]
        for i in range(df.shape[0]):
            pred = SVM.Predict(df.iloc[i,:])
            
            if df.iloc[i,-1] != pred:
                num_wrong += 1
                
    
        print('For C = ', C, '/873 \nTraining error was:', num_wrong/num_total)
            
        num_wrong = 0
        num_total = test_df.shape[0]
        for i in range(test_df.shape[0]):
            pred = SVM.Predict(test_df.iloc[i,:])
            
            if test_df.iloc[i,-1] != pred:
                num_wrong += 1
                
    
        print('\nTesting error was:', num_wrong/num_total)
        """
        print('Num alphas = ', len(SVM.alphas))
        #print(SVM.w)
    
# %% 
   
SVM = DualSVM(C = C/873, kernel = GaussianKernel)

#num_examples = 200;
SVM.BuildSVM(df.iloc[:,:])

num_wrong = 0
num_total = df.shape[0]
for i in range(df.shape[0]):
    pred = SVM.Predict(df.iloc[i,:])
    
    if df.iloc[i,-1] != pred:
        num_wrong += 1
        

print('For C = ', C, '/873 \nTraining error was:', num_wrong/num_total)
print(SVM.w)
