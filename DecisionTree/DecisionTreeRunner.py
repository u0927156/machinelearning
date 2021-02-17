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




def PrintTable(train, test):
    print('\t\t\tEntropy\t\tME\t\tGini')
    for maxDepth in range(1,7):
        EntropyTree = DecisionTree(train, maxDepth, 0)
        METree = DecisionTree(train, maxDepth, 1)
        GiniTree = DecisionTree(train, maxDepth, 2)
        print("maxDepth=%02d, %6.5f, %6.5f, %6.5f" % (maxDepth, EntropyTree.GetAccuracyLevel(test),METree.GetAccuracyLevel(test),GiniTree.GetAccuracyLevel(test)))

print('Test on Training Data')
PrintTable(trainingData, trainingData)
print('\n')
print('Test on Testing Data')
PrintTable(trainingData, testData)