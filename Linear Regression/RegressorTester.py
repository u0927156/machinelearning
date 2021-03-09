# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:23:45 2021

@author: Spencer Peterson
"""
import os
import sys
import pandas as pd
import numpy as np

EnsemblePath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Linear Regression'))
sys.path.append(EnsemblePath)
from LMSRegressor import LMSRegressor

train = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\concrete\\train.csv'
test = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\concrete\\test.csv'

df = pd.read_csv(train, header=None)

Regressor = LMSRegressor(r = 0.01)

Regressor.Regress(df, Max_Iterations = 10000)

for i in range(10):
    print('Prediction:',Regressor.Predict(df.iloc[i,:]) , 'Actual:', df.iloc[i,-1])
    
# %%
import matplotlib.pyplot as plt


fig = plt.figure()
plt.xlabel('Num Iterations')
plt.ylabel('Cost Function')
plt.plot(Regressor.CostFunction)

plt.title('Cost Function of LMS Algorithm')


FigOutAccuracy = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\LMSCostFunction.png'
plt.savefig(FigOutAccuracy, dpi=fig.dpi)

plt.show()

# %% 
test_df = pd.read_csv(test, header=None)

Predictions = np.empty(0)
for i in range(test_df.shape[0]):
    Predictions = np.append(Predictions, Regressor.Predict(test_df.iloc[i,:]))

Actual = np.array(test_df.iloc[:,-1])

CostOfTest = 1/2 * sum(np.square(Predictions-Actual))

print('The cost function with test data is:',CostOfTest)

# %% SGD regressor 

from SGDRegressor import SGDRegressor

Regressor_SGD = SGDRegressor(r = 0.011)

Regressor_SGD.Regress(df, Max_Iterations = 200)


#for i in range(10):
   # print('Prediction:',Regressor_SGD.Predict(df.iloc[i,:]) , 'Actual:', df.iloc[i,-1])
    

# %%
fig = plt.figure()
plt.xlabel('Num Updates')
plt.ylabel('Cost Function')
plt.plot(Regressor_SGD.CostFunction)

plt.title('Cost Function of SGD Algorithm')


FigOutAccuracy = 'D:\\School\\Spring 2021\\CS 6350\\Homework\\HW2\\SGDCostFunction.png'
plt.savefig(FigOutAccuracy, dpi=fig.dpi)


Predictions = np.empty(0)
for i in range(test_df.shape[0]):
    Predictions = np.append(Predictions, Regressor_SGD.Predict(test_df.iloc[i,:]))

Actual = np.array(test_df.iloc[:,-1])

CostOfTest = 1/2 * sum(np.square(Predictions-Actual))

print('The cost function with test data is:',CostOfTest)

# %% Analytical Solution
step1 = np.linalg.inv(np.dot(X_prime.T, X_prime))
step2 = np.dot(step1,X_prime.T)
analytical_solution = np.dot(step2, y.T)
