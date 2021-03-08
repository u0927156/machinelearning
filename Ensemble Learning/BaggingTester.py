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

num_bag = 40
num_trees = 40

ListOfBags = []
for bag in range(num_bag):
    
    print(bag, '/', num_bag) 
    examples = random.sample(range(df.shape[0]), 1000)
    CurrDF = df.iloc[examples,:]
    CurrBag = BaggedTree(CurrDF, CompleteDataSet=df)
    
    CurrBag.BuildNTrees(num_trees)
    
    ListOfBags.append(CurrBag)

import dill

filename = ('VarianceSession.pkl')

dill.dump_session(filename)
print('saved')
# %% 
import dill

filename = ('VarianceSession.pkl')

dill.load_session(filename)

# %% 
import dill

# copied file to different locatio to prevent overwriting
load_file_name = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\VarianceSession.pkl' 

dill.load_session(load_file_name)

# %% Bias and Variance of Each Tree

# Get all of the individual trees
Trees = []
for Bag in ListOfBags:
    Trees.append(Bag.trees[0])
    
# calculate biases
biases = []
predictions = np.zeros([num_trees, test_df.shape[0]])
for i in range(0, test_df.shape[0]):
    if i % 1000 == 0:
        print(i)
    sum_for_example = 0
    for Tree_ind in range(len(Trees)):
        Tree = Trees[Tree_ind]
        if Tree.Predict(test_df.iloc[i,:]) == 'no':
            sum_for_example += 0
            predictions[Tree_ind, i] = 0
        else:
            sum_for_example += 1
            predictions[Tree_ind, i] = 1
            
    correct_answer = 0
    if (test_df.iloc[i,-1]=='yes'):
        correct_answer = 1
    biases.append(np.square((sum_for_example/num_trees) - correct_answer))
    
Variances = []
for Tree_ind in range(len(Trees)):
    samples = predictions[Tree_ind,:]
    mean_samples = sum(samples)/len(samples)
    variance = 1/(len(samples-1)) * sum(np.square(predictions[0,:] - np.mean(predictions[0,:])))
    Variances.append(variance)

bias = sum(biases)/len(biases)
variance = sum(Variances)/len(Variances)
print('The Bias was %4.3f The variance was %4.3f. The general squared error war %4.3f' %(bias, variance, bias+variance))

# %% Bias and Variance of Each Bagging Method
  
# calculate biases
biases = []
predictions = np.zeros([num_bag, test_df.shape[0]])
for i in range(0, test_df.shape[0]):
    if i % 1000 == 0:
        print(i)
    sum_for_example = 0
    for bag_ind in range(len(ListOfBags)):
        Bag = ListOfBags[bag_ind]
        if Bag.Predict(test_df.iloc[i,:]) == 'no':
            sum_for_example += 0
            predictions[bag_ind, i] = 0
        else:
            sum_for_example += 1
            predictions[bag_ind, i] = 1
            
    correct_answer = 0
    if (test_df.iloc[i,-1]=='yes'):
        correct_answer = 1
    biases.append(np.square((sum_for_example/num_bag) - correct_answer))
    
Variances = []
for bag_ind in range(len(ListOfBags)):
    samples = predictions[bag_ind,:]
    mean_samples = sum(samples)/len(samples)
    variance = 1/(len(samples-1)) * sum(np.square(predictions[0,:] - np.mean(predictions[0,:])))
    Variances.append(variance)

bias = sum(biases)/len(biases)
variance = sum(Variances)/len(Variances)
print('The Bias was %4.3f The variance was %4.3f. The general squared error was %4.3f' %(bias, variance, bias+variance))



# %% 
