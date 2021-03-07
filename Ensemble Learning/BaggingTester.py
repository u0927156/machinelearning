# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:41:25 2021

@author: Spencer Peterson
"""

import os
import sys
import pandas as pd
import numpy as np

DecisionTreePath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'DecisionTree'))
sys.path.append(DecisionTreePath)
import TreeHelper

EnsemblePath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Ensemble Learning'))
sys.path.append(EnsemblePath)
from Bagging import BaggedTree

# %%

train = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\train.csv'
test = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\test.csv'

df = pd.read_csv(train, header=None)
TreeHelper.ProcessDataFrame(df)


Bagged = BaggedTree(df)


Bagged.BuildNTrees(1)


# %% 
test_df = pd.read_csv(test, header=None)

TreeHelper.ProcessDataFrame(test_df)
# %%

print(Bagged.Predict(test_df.iloc[102,:]))

# %% Varying Number of Trees

# Training and Testing Files
train = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\train.csv'
test = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\test.csv'


# Read into dataframe
df = pd.read_csv(train, header=None)
TreeHelper.ProcessDataFrame(df)

test_df = pd.read_csv(test, header=None)
TreeHelper.ProcessDataFrame(test_df)

# Create Bagged Predictor
Bagged = BaggedTree(df)
Bagged.BuildNTrees(1)




# %%

training_table = pd.read_csv(train, header=None)
testing_table = pd.read_csv(test, header=None)

TreeHelper.ProcessDataFrame(training_table, False)
TreeHelper.ProcessDataFrame(testing_table, False)




training_accuracy = []
testing_accuracy = []
num_trees = []


range_end = 500

# Create Bagged Predictor
Bagged = BaggedTree(df)
Bagged.BuildNTrees(1)


for i in range(0, range_end):
    print(i+1, '/', range_end)
    training_accuracy.append(Bagged.GetAccuracyLevel(training_table))
    testing_accuracy.append(Bagged.GetAccuracyLevel(testing_table))
    num_trees.append(len(Bagged.trees))
    
    
    Bagged.BuildNTrees(1)
    
# %% 
print(training_accuracy)
print(testing_accuracy)
# %% plot those results
import matplotlib.pyplot as plt


fig = plt.figure()
color ='tab:red'
plt.xlabel('Num Trees')
plt.ylabel('Error Rate Training Data')
plt.plot(num_trees, training_accuracy, color=color, label='Training')

#ax2 = ax1.twinx()


color = 'tab:blue'
#ax2.set_ylabel('Error Rate Test Data', color = color)
plt.plot(num_trees, testing_accuracy, color=color, label='Testing')

plt.legend(loc="right")
plt.title('Traing and Testing Accuracy of Bagged Trees')


FigOutAccuracy = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\BaggedAccuracy.png'
plt.savefig(FigOutAccuracy, dpi=fig.dpi)

plt.show()

# %% Bias and Variance Decomposition

import random

num_bag = 50
num_trees = 50

ListOfBags = []
for bag in range(num_bag):
    
    print(bag, '/', num_bag) 
    examples = random.sample(range(df.shape[0]), 1000)
    CurrDF = df.iloc[examples,:]
    CurrBag = BaggedTree(CurrDF, CompleteDataSet=df)
    
    CurrBag.BuildNTrees(num_trees)
    
    ListOfBags.append(CurrBag)
# %%
import dill

filename = ('VarianceSession.pkl')

dill.dump_session(filename)