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
    
    def __init__(self, df, HandleUnknowns=False, CompleteDataSet=None):
        self.__HandleUnknowns = HandleUnknowns
        self.df = df
        self.trees = []
        self.CompleteDataSet = CompleteDataSet
        
    def BuildNTrees(self, n):
        for i in range(0,n):
            self.__AppendTree()
            
    def __AppendTree(self):
        num_rows = self.df.shape[0]
        
        bootstraps = np.random.randint(0,num_rows, num_rows); # get random rows with replacement
        
        
        df_boot = self.df.iloc[bootstraps,:] # actually get the rows
        
        currTree = DecisionTree()
        
        CompleteDataSet = self.df
        if self.CompleteDataSet is not None:
            CompleteDataSet = self.CompleteDataSet
            
        currTree.BuildFromDataFrame(df_boot, num_rows, 2, None, self.__HandleUnknowns, CompleteDataSet)
        
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
        
    def GetAccuracyLevel(self, df):
        if len(self.trees) == 0:
            raise AttributeError("No trees have been constructed")
            
        count = 0
        incorrect = 0
        for i in range(0, len(df)):
            count +=1
            if self.Predict(df.iloc[i,:]) != df.iloc[i,-1]:
                incorrect+=1
            
        return incorrect/count
        
        
        
        

        