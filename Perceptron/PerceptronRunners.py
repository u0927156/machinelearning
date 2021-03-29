# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 08:38:56 2021

@author: Spencer Peterson
"""

import os
import sys
import pandas as pd
import numpy as np

PerceptronPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Perceptron'))
sys.path.append(PerceptronPath)
from Perceptron import Perceptron

train = os.path.abspath(os.path.join(PerceptronPath, 'train.csv')) # located in the peceptron file
test = os.path.abspath(os.path.join(PerceptronPath, 'test.csv'))


df = pd.read_csv(train, header=None)

# %% 
perceiver = Perceptron()

perceiver.BuildPerceptron(df)


#for i in range(10):
#   print('Prediction:',perceiver.Predict(df.iloc[i,:]) , 'Actual:', df.iloc[i,-1])
    


test_df = pd.read_csv(test, header=None)

num_wrong = 0
num_total = test_df.shape[0]
for i in range(test_df.shape[0]):
    pred = perceiver.Predict(test_df.iloc[i,:])
    
    if test_df.iloc[i,-1] != pred:
        num_wrong += 1
        
print('The weight vector was: $', perceiver.w, '$.')
print('Testing error was:', num_wrong/num_total)

# %% voted perceptron
from VotedPerceptron import VotedPerceptron

voter = VotedPerceptron()

voter.BuildPerceptron(df)

#for i in range(10):
#   print('Prediction:',voter.Predict(df.iloc[i,:]) , 'Actual:', df.iloc[i,-1])
    
 
num_wrong = 0
num_total = test_df.shape[0]
for i in range(test_df.shape[0]):
    pred = voter.Predict(test_df.iloc[i,:])
    
    if test_df.iloc[i,-1] != pred:
        num_wrong += 1
        
print('Testing error was:', num_wrong/num_total)


# %% 

for i in range(voter.w.shape[0]):
    w = voter.w[i]
    c = voter.C[i]
    print('%4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %2d\\\\ \\hline' % (w[0], w[1], w[2], w[3], w[4], c))
    
# %% Averager

from AveragedPerceptron import AveragedPerceptron

averager = AveragedPerceptron()

averager.BuildPerceptron(df)





#for i in range(10):
 #   print('Prediction:',perceiver.Predict(df.iloc[i,:]) , 'Actual:', df.iloc[i,-1])
    
     
    

test_df = pd.read_csv(test, header=None)

num_wrong = 0
num_total = test_df.shape[0]
for i in range(test_df.shape[0]):
    pred = averager.Predict(test_df.iloc[i,:])
    
    if test_df.iloc[i,-1] != pred:
        num_wrong += 1
        
print('The weight vector was: $', averager.a, '$.')
print('Testing error was:', num_wrong/num_total)
    