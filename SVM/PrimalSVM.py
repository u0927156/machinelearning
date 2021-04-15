# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:54:37 2021

@author: Spencer Peterson
"""
import pandas as pd
import numpy as np


class PrimalSVM:
    """
    Binary learner perceptron. Takes a dataframe and builds a classifier.
    """
    
    def __init__(self, gamma = 0.010, C = 100/873,  epochs = 10, learning_schedule=0, d=1):
        """
        Constructor for perceptron 

        Parameters
        ----------
        r : float, optional
            The learning rate for the algorithm. The default is 0.010.
        epochs : int, optional
            The number of times to cycle through the training data. The default is 10.

        Returns
        -------
        None.

        """
        self.w = None
        self.gamma = gamma
        self.C = C
        self.CostFunction = np.empty(0)
        self.epochs = epochs
        self.learning_schedule = learning_schedule
        self.d = d
        
        
    def BuildSVM(self, df):
        """
        Builds the classifier based on the data

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame used to construct the SVM. Must contain correct classification in the final row. 
            The correct classification must be 0 or 1

        Returns
        -------
        None.

        """
        # Sets all 0s to -1 for the math to work.
        df = df.copy()
        df.loc[df[df.columns[-1]] == 0, df.columns[-1]] = -1
        
       
        # initialize the w with an extra term for the bias 
        self.w = np.zeros(df.shape[1])
        
        N = df.shape[0]
        
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
            
            t = 0
            # for each training example in df
            for j in range(X_prime.shape[0]):
                
                if self.learning_schedule == 0:
                    gamma_curr = self.gamma / (1 + (self.gamma * t / self.d))
                else:
                    gamma_curr = self.gamma / (1 + t)
                    
                t += 1
                    
                if y[j] * np.dot(self.w, X_prime[j]) <= 1:
                    delta_w = -self.w * gamma_curr  + gamma_curr * self.C * N *y[j] * X_prime[j]
                    
                    
                    # check for convergence
                    if np.all(np.absolute(delta_w) < np.finfo(float).eps):
                        return
                    
                    new_w = self.w + delta_w
                else:
                    new_w = (1-gamma_curr) * self.w

                self.w = new_w
                
                
                
                

            
        
    def Predict(self, row):
        """
        Gets prediction based on constructed perceptron

        Parameters
        ----------
        row : pandas.Series
            The row of data that is used to predict. The code assumes the correct answer is in the final column.

        Returns
        -------
        int
            The prediction. Either 1 or 0.

        """
        # input is  assumed to have output on it
        x_in = np.array(row[0:-1])
        x_in_prime = np.insert(x_in, 0, 1)
        
        prediction = np.dot(self.w, x_in_prime)
        
        if prediction >= 0:
            return 1
        else:
            return 0