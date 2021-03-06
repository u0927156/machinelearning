# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:37:06 2021

@author: Spencer Peterson
"""


# %%
import os
import sys
import pandas as pd
import numpy as np
import TreeHelper

# %%
AdaboostPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Ensemble Learning'))
sys.path.append(AdaboostPath)
from AdaBoost import AdaBoostTree


boost = AdaBoostTree()

train = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\train.csv'
test = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\test.csv'

boost.BuildAdaBoost(train, 1)




# %%

training_table = pd.read_csv(train, header=None)
testing_table = pd.read_csv(test, header=None)

TreeHelper.ProcessDataFrame(training_table, False)
TreeHelper.ProcessDataFrame(testing_table, False)




training_accuracy = []
testing_accuracy = []
num_trees = []
boost = AdaBoostTree()

train = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\train.csv'
test = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\test.csv'

boost.BuildAdaBoost(train, 1)

range_end = 500

for i in range(0, range_end):
    print(i+1, '/', range_end)
    training_accuracy.append(boost.GetAccuracyLevel(training_table))
    testing_accuracy.append(boost.GetAccuracyLevel(testing_table))
    num_trees.append(len(boost.trees))
    
    
    boost.AppendTree()

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
plt.title('Traing and Testing Accuracy of AdaBoost')


FigOutAccuracy = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\AdaBoostAccuracy.png'
plt.savefig(FigOutAccuracy, dpi=fig.dpi)

plt.show()

# %% Go through individual trees now
tree_num = []
tree_training_acc = []
tree_testing_acc = []
range_end = len(boost.trees)
for i in range(0, range_end):
    currTree = boost.trees[i]
    print(i+1, '/', range_end)
    tree_num.append(i)
    
    tree_training_acc.append(currTree.GetAccuracyLevel(train))
    tree_testing_acc.append(currTree.GetAccuracyLevel(test))
    
# %%
fig = plt.figure()
color ='tab:red'
plt.xlabel('Num Trees')
plt.ylabel('Error Rate Training Data')
plt.plot(tree_num, tree_training_acc, '--k', label='Training')

#ax2 = ax1.twinx()


color = 'tab:blue'
#ax2.set_ylabel('Error Rate Test Data', color = color)
plt.plot(tree_num, tree_training_acc, '.r', label='Testing')

plt.legend(loc="upper right")
plt.title('Traing and Testing Accuracy of AdaBoost')


TreeAccuracy = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\TreeAccuracy.png'
plt.savefig(TreeAccuracy, dpi=fig.dpi)

plt.show()

# %% zoom in on figure

fig = plt.figure()
color ='tab:red'
plt.xlabel('Num Trees')
plt.ylabel('Error Rate Training Data')
plt.plot(tree_num, tree_training_acc, '--k', label='Training')

#ax2 = ax1.twinx()


color = 'tab:blue'
#ax2.set_ylabel('Error Rate Test Data', color = color)
plt.plot(tree_num, tree_training_acc, '.r', label='Testing')

plt.legend(loc="upper right")
plt.title('Traing and Testing Accuracy of AdaBoost')

plt.xlim((0, 50))

TreeZoomedAccuracy = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\TreeAccuracyZoomed.png'
plt.savefig(TreeZoomedAccuracy, dpi=fig.dpi)

plt.show()

# %% adaboost mini
# smaller training set
mini_train = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\mini_train.csv'
test = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\bank\\test.csv'

miniboost = AdaBoostTree()
miniboost.BuildAdaBoost(mini_train, 1)

# %% 
miniboost.AppendTree()

# %%
miniboost.AppendTree()
miniboost.AppendTree()
