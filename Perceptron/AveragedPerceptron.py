# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:22:08 2021

@author: Spencer Peterson
"""

import pandas as pd
import numpy as np

class AveragedPerceptron:
    """
    Implementation of Voted Perceptron Algorithm
    """
    def __init__(self, r = 0.010, epochs = 10):
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
        self.a = None
        self.r = r
        self.CostFunction = np.empty(0)
        self.epochs = epochs
        
        
    def BuildPerceptron(self, df):
        """
        Builds the classifier based on the data

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame used to construct the perceptron. Must contain correct classification in the final row. 
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
        self.a = np.zeros(df.shape[1])
        
        
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
                
                if y[j] * np.dot(X_prime[j], self.w) <= 0:
                    new_w = self.w + self.r*y[j] * X_prime[j]
                    self.w = new_w
                self.a = self.a + self.w

                
                
                
                

            
        
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
        
        prediction = np.dot(self.a, x_in_prime)
        
        if prediction >= 0:
            return 1
        else:
            return 0
        