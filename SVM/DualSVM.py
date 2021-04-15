# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:06:40 2021

@author: Spencer Peterson
"""
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import os.path
import sys
import pandas as pd
import numpy as np

SVMPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'SVM'))
sys.path.append(SVMPath)


train = os.path.abspath(os.path.join(SVMPath, 'train.csv')) # located in the peceptron file
test = os.path.abspath(os.path.join(SVMPath, 'test.csv'))
    
    
df = pd.read_csv(train, header=None)

class DualSVM:

    def __init__(self, C = 100/873, kernel = np.dot):
        self.C = C
        self.w = None
        self.H = None
        self.x = None
        self.y = None
        self.kernel = kernel
    
    def to_minimize(self, alphas):
        return 1/2 * np.dot(alphas.T, np.dot(self.H, alphas)) - np.sum(alphas)
        
    def derivatives(self, alphas):
        return np.dot(alphas.T, self.H - np.ones(alphas.shape[0]))
                      
    def H_matrix(self, X, Y):
        H = np.zeros((X.shape[0], X.shape[0]))
        for row in range(X.shape[0]):
            for col in range(X.shape[0]):
                H[row,col] = self.kernel(X[row,:],X[col,:])*Y[row]*Y[col]
        return H
    
    
    def BuildSVM(self, df):

        # Clean up data        
        df.loc[df[df.columns[-1]] == 0, df.columns[-1]] = -1

        x_temp = np.array(df.iloc[:,0:-1])
        ones = np.ones(len(x_temp))
        self.x = np.insert(x_temp, 0, ones, axis = 1)
        self.y = np.array(df.iloc[:,-1])
        
        
        # Make a matrix with the kernel trick
        self.H = self.H_matrix(self.x,self.y)
        
        # Parameters for minimize
        A = self.y[:]
        
        cons = {'type' : 'eq',
              'fun' : lambda alphas: np.dot(A,alphas),
              'jac' : lambda alphas: A}
        
        bounds = [(0,self.C)]*self.x.shape[0] # alpha has to be greater than 0 and less than C
        
        # Find the alphas
        sol = minimize(self.to_minimize, np.random.rand(self.x.shape[0]), jac=self.derivatives, 
                       constraints=cons, bounds = bounds)
        
        self.non_zero_alphas = sol.x>0
        self.alphas = sol.x[self.non_zero_alphas]
        
        self.w = np.array((self.y[self.non_zero_alphas]*self.x[self.non_zero_alphas].T*self.alphas).sum(axis=1))
        
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
        self.x_in_prime = x_in_prime
        return np.sign(sum((self.alphas * self.y[self.non_zero_alphas]*self.kernel(self.x[self.non_zero_alphas], x_in_prime))))
        