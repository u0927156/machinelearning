# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:20:11 2021

@author: Spencer Peterson
"""

import pandas as pd
import numpy as np

class LMSRegressor:
    
    def __init__(self, r = 0.010):
        self.w = None
        self.r = r
        self.CostFunction = np.empty(0)
        
    def Regress(self, df, Max_Iterations=100):
        
        # y is last column
        y = np.array(df.iloc[:,-1])
        
        # x is all columns except last one.
        X = np.array(df.iloc[:,:-1])
        
        # insert a one for the bias term. 
        ones = np.ones(len(X))
        X_prime = np.insert(X, 0, ones, axis = 1)
        
        # initialize w
        self.w = np.zeros(X_prime.shape[1])
        
        for i in range(Max_Iterations):
            #print(self.w)
            predictions = np.dot(X_prime, self.w)
            
            cost = 1/2 * sum(np.square((y-predictions)))
            
            self.CostFunction = np.append(self.CostFunction, cost)
            
            grad = np.dot(X_prime.T, predictions-y)
            
            new_w = self.w - self.r*grad
            
            if(np.all(np.isclose(new_w,self.w,atol=1e-06))):
                self.w = new_w
                print('Converged after ', i, 'iterations')
                break
            
            self.w = new_w
            
        
    def Predict(self, row):
        # input is  assumed to have output on it
        x_in = np.array(row[0:-1])
        x_in_prime = np.insert(x_in, 0, 1)
        
        prediction = np.dot(self.w, x_in_prime)
        return prediction