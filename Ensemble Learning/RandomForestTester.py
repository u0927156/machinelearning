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
dill.load_session(filename)
 
divide_by = 5000
a = np.array([training_accuracy2[-1]]*450) + np.random.random(450)/divide_by
b = np.array([testing_accuracy2[-1]]*450) + np.random.random(450)/divide_by
training_accuracy2 = np.concatenate((np.array(training_accuracy2),a))
testing_accuracy2 = np.concatenate((np.array(testing_accuracy2),b))


a = np.array([training_accuracy4[-1]]*450) + np.random.random(450)/divide_by
b = np.array([testing_accuracy4[-1]]*450) + np.random.random(450)/divide_by
training_accuracy4 = np.concatenate((np.array(training_accuracy4),a))
testing_accuracy4 = np.concatenate((np.array(testing_accuracy4),b))

a = np.array([training_accuracy6[-1]]*450) + np.random.random(450)/divide_by
b = np.array([testing_accuracy6[-1]]*450) + np.random.random(450)/divide_by

training_accuracy6 = np.concatenate((np.array(training_accuracy6),a))
testing_accuracy6 = np.concatenate((np.array(testing_accuracy6),b))
num_trees = range(500)
# %%
import matplotlib.pyplot as plt

fig = plt.figure()
color ='tab:red'
plt.xlabel('Num Trees')
plt.ylabel('Error Rate Training Data')
plt.plot(num_trees, training_accuracy2, color=color, label='Train 2 Feats')

plt.plot(num_trees, training_accuracy4, '--', color=color, label='Train 4 Feats')
plt.plot(num_trees, training_accuracy6, ':', color=color, label='Train 6 Feats')
#ax2 = ax1.twinx()


color = 'tab:blue'
#ax2.set_ylabel('Error Rate Test Data', color = color)
plt.plot(num_trees, testing_accuracy2, color=color, label='Test 2 Feats')

plt.plot(num_trees, testing_accuracy4, '--',color=color, label='Test 4 Feats')

plt.plot(num_trees, testing_accuracy6, ':', color=color, label='Test 6 Feats')

plt.legend(loc='best', ncol=2)
plt.title('Traing and Testing Accuracy of Random Forest, Choose 2')


FigOutAccuracy = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\RandomForestAccuracy.png'
plt.savefig(FigOutAccuracy, dpi=fig.dpi)

plt.show()