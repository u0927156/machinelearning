# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:25:58 2021

@author: Spencer Peterson
"""

# do all the imports
import sys
import os
import numpy as np
import pandas as pd

DecisionTreePath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'DecisionTree'))
sys.path.append(DecisionTreePath)
from DecisionTree import DecisionTree

class BaggedTree:
    
    def __init__(self, df, HandleUnknowns=False):
        self.__HandleUnknowns = HandleUnknowns
        self.df = df
        self.trees = []
        
    def BuildNTrees(self, n):
        for i in range(0,n):
            self.__AppendTree()
            
    def __AppendTree(self):
        num_rows = self.df.shape[0]
        
        bootstraps = np.random.randing(0,num_rows, num_rows); # get random rows with replacement
        
        df_boot = self.df.iloc[:, bootstraps] # actually get the 
        
        currTree = DecisionTree()
        
        currTree.BuildFromDataFrame(df_boot, num_rows, 2, None, self.__HandleUnkowns)
        
        self.trees.append(currTree)
        
    def Predict(self, row):
        if len(self.trees) == 0:
             raise AttributeError("No trees have been constructed")
             
             
        predictions = {}
        
        for i in range(0, len(self.trees)):
            currTree = self.trees[i]
            
            currPred = currTree.Predict(row)
            
            if currPred in predictions:
                predictions[currPred] += 1
            else:
                predictions[currPred] = 1
                

        predWeight = -10000
        
        
        PredictionToReturn = ''
        for key in predictions:
            if predictions[key] > predWeight:
                predWeight = predictions[key]
                PredictionToReturn = key
                
        return PredictionToReturn
        
        
        
        
        

        