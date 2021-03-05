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
        self.D = None;
        
    def BuildAdaBoost(self, filename, iterations):
        df = pd.read_csv(filename)
        
        df_rows = df.shape[0]
        D_0 = [1/df_rows]*df_rows
        
        
        