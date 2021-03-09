# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:26:02 2021

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
from RandomForest import RandomForest

# %%

train = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\train.csv'
test = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\test.csv'

df = pd.read_csv(train, header=None)
TreeHelper.ProcessDataFrame(df)


forest = RandomForest(df, 2)


forest.BuildNTrees(1)



# %% 
test_df = pd.read_csv(test, header=None)

TreeHelper.ProcessDataFrame(test_df)
# %%

print(forest.Predict(test_df.iloc[102,:]))

# %% Varying Number of Trees

train = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\train.csv'
test = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\test.csv'

training_table = pd.read_csv(train, header=None)
testing_table = pd.read_csv(test, header=None)

TreeHelper.ProcessDataFrame(training_table, False)
TreeHelper.ProcessDataFrame(testing_table, False)




training_accuracy2 = []
testing_accuracy2 = []
num_trees = []


training_accuracy4 = []
testing_accuracy4 = []


training_accuracy6 = []
testing_accuracy6 = []

range_end = 50

# Create Bagged Predictor
forest2 = RandomForest(training_table, 2)
forest2.BuildNTrees(1)


forest4 = RandomForest(training_table, 4)
forest4.BuildNTrees(1)


forest6 = RandomForest(training_table, 6)
forest6.BuildNTrees(1)

for i in range(0, range_end):
    print(i+1, '/', range_end)
    training_accuracy2.append(forest2.GetAccuracyLevel(training_table))
    testing_accuracy2.append(forest2.GetAccuracyLevel(testing_table))
    forest2.BuildNTrees(1)
    
    training_accuracy4.append(forest4.GetAccuracyLevel(training_table))
    testing_accuracy4.append(forest4.GetAccuracyLevel(testing_table))
    forest4.BuildNTrees(1)

    training_accuracy6.append(forest6.GetAccuracyLevel(training_table))
    testing_accuracy6.append(forest6.GetAccuracyLevel(testing_table))
    forest6.BuildNTrees(1)
    
    num_trees.append(len(forest.trees))
    
    
    forest.BuildNTrees(1)
    
import dill

filename = ('Forest500.pkl')

dill.dump_session(filename)
print('saved')

# %%
import matplotlib.pyplot as plt

fig = plt.figure()
color ='tab:red'
plt.xlabel('Num Trees')
plt.ylabel('Error Rate Training Data')
plt.plot(num_trees, training_accuracy2,  label='Train 2 Feats')

plt.plot(num_trees, training_accuracy4, '--', label='Train 4 Feats')
plt.plot(num_trees, training_accuracy6, ':',  label='Train 6 Feats')
#ax2 = ax1.twinx()



#ax2.set_ylabel('Error Rate Test Data', color = color)
plt.plot(num_trees, testing_accuracy2, label='Test 2 Feats')

plt.plot(num_trees, testing_accuracy4, '--', label='Test 4 Feats')

plt.plot(num_trees, testing_accuracy6, ':', label='Test 6 Feats')

plt.legend(loc='best', ncol=2)
plt.title('Traing and Testing Accuracy of Random Forest, Choose 2')


FigOutAccuracy = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\RandomForestAccuracy.png'
plt.savefig(FigOutAccuracy, dpi=fig.dpi)

plt.show()


# %% Bias and Variance Decomposition

import random

num_bag = 100
num_trees = 1000

ListOfForests = []
for bag in range(num_bag):
    
    print(bag, '/', num_bag) 
    examples = random.sample(range(df.shape[0]), 1000)
    CurrDF = training_table.iloc[examples,:]
    CurrForest = RandomForest(CurrDF, 4, CompleteDataSet=training_table)
    
    CurrForest.BuildNTrees(num_trees)
    
    ListOfForests.append(CurrForest)

import dill

filename = ('VarianceSessionForest.pkl')

dill.dump_session(filename)
print('saved')
# %% Bias and Variance of Each Tree

# Get all of the individual trees
Trees = []
for Forest in ListOfForests:
    Trees.append(Forest.trees[0])
    
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
print('The bias was %4.3f The variance was %4.3f. The general squared error was %4.3f' %(bias, variance, bias+variance))

# %% Bias and Variance of Each Forest Method
  
# calculate biases
biases = []
predictions = np.zeros([num_bag, test_df.shape[0]])
for i in range(0, test_df.shape[0]):
    if i % 1000 == 0:
        print(i)
    sum_for_example = 0
    for forest_ind in range(len(ListOfForests)):
        Forest = ListOfForests[forest_ind]
        if Forest.Predict(test_df.iloc[i,:]) == 'no':
            sum_for_example += 0
            predictions[forest_ind, i] = 0
        else:
            sum_for_example += 1
            predictions[forest_ind, i] = 1
            
    correct_answer = 0
    if (test_df.iloc[i,-1]=='yes'):
        correct_answer = 1
    biases.append(np.square((sum_for_example/num_bag) - correct_answer))
    
Variances = []
for forest_ind in range(len(ListOfForests)):
    samples = predictions[forest_ind,:]
    mean_samples = sum(samples)/len(samples)
    variance = 1/(len(samples-1)) * sum(np.square(predictions[0,:] - np.mean(predictions[0,:])))
    Variances.append(variance)

bias = sum(biases)/len(biases)
variance = sum(Variances)/len(Variances)
print('For the Random Forest, the bias was %4.3f The variance was %4.3f. The general squared error was %4.3f' %(bias, variance, bias+variance))


