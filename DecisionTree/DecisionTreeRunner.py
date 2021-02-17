# -*- coding: utf-8 -*-
"""
Script for running experiments for HW1

Created on Wed Feb 17 10:16:48 2021

@author: Spencer Peterson
"""
from DecisionTree import DecisionTree

import os
import numpy as np
import pandas as pd


trainingData = 'car/train.csv'

FirstTree = DecisionTree(trainingData, 1, 0)

testData =  'car/test.csv'




def PrintTable(train, test, rangeEnd):
    print('\t\t\tEntropy\t\tME\t\tGini')
    for maxDepth in range(1,rangeEnd+1):
        EntropyTree = DecisionTree(train, maxDepth, 0)
        METree = DecisionTree(train, maxDepth, 1)
        GiniTree = DecisionTree(train, maxDepth, 2)
        print("%2d & %5.4f & %5.4f & %5.4f \\\\ \\hline" % (maxDepth, EntropyTree.GetAccuracyLevel(test),METree.GetAccuracyLevel(test),GiniTree.GetAccuracyLevel(test)))

print('Test on Training Data')
PrintTable(trainingData, trainingData,6)
print('\n')
print('Test on Testing Data')
PrintTable(trainingData, testData,6)

# %%
bankTrainingData = 'bank/train.csv'
bankTestingData = 'bank/test.csv'
print('Bank Test on Training Data, count Unknown as Value\n')
PrintTable(bankTrainingData, bankTrainingData, 16)

# %%
print('Bank Test on Testing Data, count Unknown as Value')
PrintTable(bankTrainingData, bankTestingData, 16 

# %%
