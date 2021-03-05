# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 08:48:13 2021

@author: Spencer Peterson
"""


import sys
import os
import numpy as np
import pandas as pd

DecisionTreePath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'DecisionTree'))
sys.path.append(DecisionTreePath)
from DecisionTree import DecisionTree



class AdaBoostTree:
    
    def __init__(self):
        self.trees = []
        self.alphas = []
        self.D = None;
        self.df = None
    def BuildAdaBoost(self, filename, T):
        df = pd.read_csv(filename, header=None)
        
        df_rows = df.shape[0]
        self.D = np.array([1/df_rows]*df_rows)
        
        self.df = df
        for t in range(0,T):
            self.__AddNewTree()
            
            
    def __AddNewTree(self):
        df = self.df
        currTree = DecisionTree()
        currTree.BuildFromDataFrame(df, 1, 2, self.D, True)
        
        
        
        predictions = self.__GetPredictions(df, currTree)
        error = self.__CalculateTreeError(df, predictions)

        print(error)
        alpha = self.__CalculateAlpha(error)
        self.alphas.append(alpha)
        
        self.__UpdateD(df, predictions, alpha)
        

        self.trees.append(currTree)
        
        
            
    def __CalculateTreeError(self, df, predictions):
        error = 0
        for i in range(0, len(df)):
            if predictions[i] != df.iloc[i,-1]:
                error += self.D[i]
        
        return error
                
    def __GetPredictions(self, df, tree):
        predictions = []
        for i in range(0, len(df)):
            predictions.append(tree.Predict(df.iloc[i,:]))
                               
        return predictions  


    def __CalculateAlpha(self, error):
        return 1/2 * np.log((1-error)/error)
    
    
    def __UpdateD(self, df, predictions, alpha):
        
        for i in range(0, len(df)):
            # make correct predictions weigh less
            if predictions[i] == df.iloc[i,-1]:
                self.D[i] = self.D[i] * np.exp(-alpha)
            # make incorrect predictions weigh more
            else:
                self.D[i] = self.D[i] * np.exp(alpha)
        # normalize vector to keep it equal to one
        self.D = self.D / sum(self.D)
        
        
    def Predict(self, row):
        predictions = {}
        
        for i in range(0, len(self.trees)):
            currTree = self.trees[i]
            
            currPred = currTree.Predict(row)
            
            if currPred in predictions:
                predictions[currPred] += self.alphas[i]
            else:
                predictions[currPred] = self.alphas[i]
                

        predWeight = -10000
        
        PredictionToReturn = ''
        for key in predictions:
            if predictions[key] > predWeight:
                predWeight = predictions[key]
                PredictionToReturn = key
                
        return PredictionToReturn
            
                
            
        
            
        
        
        
        
        