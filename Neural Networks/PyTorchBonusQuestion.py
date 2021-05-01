# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:59:44 2021

@author: Spencer Peterson
"""
import torch
import numpy as np
import pandas as pd

import os
import sys


import matplotlib.pyplot as plt

device = torch.device('cpu')

# %% define network class
from torch.nn import Module, ReLU, Linear, ModuleList, Parameter, Tanh, init

class SelfLinear(Module):
    def __init__(self, n_in, n_out):
        super(SelfLinear, self).__init__()
        self.weight = Parameter(torch.tensor(np.random.normal(size=(n_out, n_in))))
        self.bias = Parameter(torch.tensor(np.random.rand(1, n_out)))
        
    def forward(self, X):
        return X @ self.weight.T + self.bias
    
    
class Net(Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.act = Tanh()
        self.layers = layers
        
        self.fcs = ModuleList()
        
        for i in range(len(self.layers)-1):
            #print(self.layers[i], self.layers[i+1])
            layer = SelfLinear(self.layers[i], self.layers[i+1])
            #init.kaiming_normal_(layer.weight)
            init.xavier_normal_(layer.weight)
            self.fcs.append(layer)

    def forward(self, X):
        for fc in self.fcs[:-1]:
            X = fc(X)
            X = self.act(X)
        X = self.fcs[-1](X)
        return X
    
class NetRelu(Module):
    def __init__(self, layers):
        super(NetRelu, self).__init__()
        self.act = ReLU()
        self.layers = layers
        
        self.fcs = ModuleList()
        
        for i in range(len(self.layers)-1):
            #print(self.layers[i], self.layers[i+1])
            layer = SelfLinear(self.layers[i], self.layers[i+1])
            init.kaiming_normal_(layer.weight)
            #init.xavier_normal_(layer.weight)
            self.fcs.append(layer)

    def forward(self, X):
        for fc in self.fcs[:-1]:
            X = fc(X)
            X = self.act(X)
        X = self.fcs[-1](X)
        return X
    
     

# %% load data
PerceptronPath = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'Perceptron'))

sys.path.append(PerceptronPath)




train = os.path.abspath(os.path.join(PerceptronPath, 'train.csv')) # located in the peceptron file
test = os.path.abspath(os.path.join(PerceptronPath, 'test.csv'))



# %% 

df = pd.read_csv(train, header=None)

#df.loc[df[df.columns[-1]] == 0, df.columns[-1]] = -1
x = np.array(df.iloc[:,:-1])
y = np.array(df.iloc[:,-1])

y_orig = y
#x = x / np.linalg.norm(x)
x = torch.tensor(x, device = device).double()
y= torch.tensor(y, device = device).double()

# %% 
x = torch.tensor(np.random.rand(100, 4))
y = torch.tensor(np.random.rand(100, 1))
# %% Tanh
depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
for depth in depths:
    for width in widths:
        
        layers = [4]
        layers += ([width]*depth)
        layers += [1]
        
        #print()
        #print('Depth = %d, width = %d' % (depth, width))
        #print()
        
        
        layers = [4,10, 10, 1]
        model = Net(layers)
        
        #model.apply(initialize_weights)
        #print(list(model.parameters()))
        
        loss = torch.nn.MSELoss(size_average=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(1,501):
            optimizer.zero_grad()
            
            y_pred = model(x)
            L = loss(y_pred, y)
            
            #print('Epoch: %d, Loss = %3f' % (epoch, L))
            
            L.backward()
            
            optimizer.step()
            
        
        
        output = model(x).detach().numpy()
        
        output = output.round()
        #output[output<0] = 0
        #output[output>0] = 1
        
        
        num_wrong = 0
        
        num_total = len(output)
        for i in range(len(output)):
            pred = output[i]
            
            if y_orig[i] != pred:
                num_wrong += 1
        
        #print('Training error was: %.3f' % (num_wrong/num_total))
        
        train_error = (num_wrong/num_total)
        test_df = pd.read_csv(test, header=None)
        
        
        x_test = np.array(test_df.iloc[:,:-1])
        
        #x = x / np.linalg.norm(x)
        x_test = torch.tensor(x_test, device = device)
        
        
        
        output = model(x_test).detach().numpy()
        output = output.round()
        
        
        num_wrong = 0
        num_total = len(output)
        
        
        for i in range(len(output)):
            pred = output[i]
            
            expected = test_df.iloc[i,-1] 
            #print(pred, test_df.iloc[i,-1])
            if expected != pred:
                
                num_wrong += 1
                
        #print('Testing error was: %.3f \\\\' % (num_wrong/num_total))
        test_error = (num_wrong/num_total)
        
        print('%d & %d & %.3f & %.3f \\hline \\\\' % (depth, width, train_error, test_error))
        
# %% Relu
depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
for depth in depths:
    for width in widths:
        
        layers = [4]
        layers += ([width]*depth)
        layers += [1]
        
        #print()
        #print('Depth = %d, width = %d' % (depth, width))
        #print()
        
        
        layers = [4,10, 10, 1]
        model = NetRelu(layers)
        
        #model.apply(initialize_weights)
        #print(list(model.parameters()))
        
        loss = torch.nn.MSELoss(size_average=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(1,501):
            optimizer.zero_grad()
            
            y_pred = model(x)
            L = loss(y_pred, y)
            
            #print('Epoch: %d, Loss = %3f' % (epoch, L))
            
            L.backward()
            
            optimizer.step()
            
        
        
        output = model(x).detach().numpy()
        
        output = output.round()
        output[output<0] = 0
        output[output>0] = 1
        
        
        num_wrong = 0
        
        num_total = len(output)
        for i in range(len(output)):
            pred = output[i]
            
            if y_orig[i] != pred:
                num_wrong += 1
        
        #print('Training error was: %.3f' % (num_wrong/num_total))
        
        train_error = (num_wrong/num_total)
        test_df = pd.read_csv(test, header=None)
        
        
        x_test = np.array(test_df.iloc[:,:-1])
        
        #x = x / np.linalg.norm(x)
        x_test = torch.tensor(x_test, device = device)
        
        
        
        output = model(x_test).detach().numpy()
        output = output.round()
        output[output<0] = 0
        output[output>0] = 1
        
        num_wrong = 0
        num_total = len(output)
        
        
        for i in range(len(output)):
            pred = output[i]
            
            expected = test_df.iloc[i,-1] 
            #print(pred, test_df.iloc[i,-1])
            if expected != pred:
                
                num_wrong += 1
                
        #print('Testing error was: %.3f \\\\' % (num_wrong/num_total))
        test_error = (num_wrong/num_total)
        
        print('%d & %d & %.3f & %.3f \\hline \\\\' % (depth, width, train_error, test_error))