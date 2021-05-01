# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:51:00 2021

@author: Spencer Peterson
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 08:22:52 2021

@author: Spencer Peterson
"""

import pandas as pd
import numpy as np

class MLERegressor:
    
    def __init__(self, epochs = 100, var = 1, gamma_0 = 1, d = 1):
        self.epochs = epochs
        self.var = var
        self.w = None 
        self.gamma_0 = gamma_0
        self.d = d
    
        
       
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    
    
    def buildRegressor(self, df):
        
        # Sets all 0s to -1 for the math to work.
        df = df.copy()
        df.loc[df[df.columns[-1]] == 0, df.columns[-1]] = -1
        
       
        # initialize the w with an extra term for the bias 
        self.w = np.zeros(df.shape[1])
       
        # set up learning rate counter
        t = 0
       
        for i in range(self.epochs):
        
            # shuffle the data
            shuffled_df = df.sample(frac=1)
            
            # y is last column
            y = np.array(shuffled_df.iloc[:,-1])
            
            # x is all columns except last one.
            X = np.array(shuffled_df.iloc[:,:-1])
            
            # insert a one for the bias term
            ones = np.ones(len(X))
            X_prime = np.insert(X, 0, ones, axis = 1)
            
            # for each training example in df
            for j in range(X_prime.shape[0]):
                curr_y = y[j]
                curr_x = X_prime[j]
   
                
                pred =  self.sigmoid(np.dot(self.w, curr_x))
                gradient = pred * (1-pred) * (curr_y - pred) * curr_x
                #print(gradient)
                learning_rate = self.gamma_0 / (1 + (self.gamma_0 / self.d) * t)
                
                self.w = self.w + learning_rate * gradient
                
               
                # increment learning rate counter
                t += 1 

        
        
    def Predict(self, row):

        # input is  assumed to have output on it
         x_in = np.array(row[0:-1])
         x_in_prime = np.insert(x_in, 0, 1)
         
         prediction = self.sigmoid(np.dot(self.w, x_in_prime))
         
         if prediction < 0.5:
             return 0
         else:
             return 1
       