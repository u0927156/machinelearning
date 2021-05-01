# -*- coding: utf-8 -*-
"""
Created on Sat May  1 08:05:01 2021

@author: Spencer Peterson
"""
import os
import sys
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt


PerceptronPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Perceptron'))
NetworkPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Neural Networks'))
sys.path.append(PerceptronPath)
sys.path.append(NetworkPath)



train = os.path.abspath(os.path.join(PerceptronPath, 'train.csv')) # located in the peceptron file
test = os.path.abspath(os.path.join(PerceptronPath, 'test.csv'))

# %% 
from NeuralNetwork import NeuralNetwork
df = pd.read_csv(train, header=None)

ds = [.1, .5, 1, 10]
gammas = [1, .5, .25, .1, .05, .01]
widths = [5, 10, 25, 50, 100]

for width in widths:
    net = NeuralNetwork(epochs = 201, gamma_0 = 0.5, d = 50000, print_error = 0, all_zeros = True)
    print('For width:', width)
    print()
    #print(width, net.d, net.gamma_0)
    net.buildNetwork(df)
    
    
    
    num_wrong = 0
    num_total = df.shape[0]
    for i in range(df.shape[0]):
        pred = net.Predict(df.iloc[i,:])
        
        if df.iloc[i,-1] != pred:
            num_wrong += 1
    
    
    print('Training error was: %.3f' % (num_wrong/num_total))
    
    test_df = pd.read_csv(test, header=None)
    
    num_wrong = 0
    num_total = test_df.shape[0]
    for i in range(test_df.shape[0]):
        pred = net.Predict(test_df.iloc[i,:])
        
        #print(pred, test_df.iloc[i,-1])
        if test_df.iloc[i,-1] != pred:
            num_wrong += 1
            
    print('Testing error was: %.3f \\\\' % (num_wrong/num_total))
    