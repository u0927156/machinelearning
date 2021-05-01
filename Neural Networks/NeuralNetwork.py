# -*- coding: utf-8 -*-
"""
Created on Sat May  1 07:48:03 2021

@author: Spencer Peterson
"""

import pandas as pd
import numpy as np
from scipy.special import expit


class NeuralNetwork:
    
    def __init__(self, epochs = 100, gamma_0 = 1, d = 1, width = 5, print_error=0, all_zeros = False):
        self.epochs = epochs
        self.network = []
        self.gamma_0 = gamma_0
        self.d = d
        self.width = width
        self.t = 0
        self.print_error = print_error
        self.all_zeros = all_zeros
        
    def buildNetwork(self, df):
        # copy to ensure original dataframe isn't changed
        df = df.copy()
        
        # construct the network nodes and shape
        self.__constructNetwork(df)
        
        # train the network
        for i in range(self.epochs):
        
            # shuffle the data
            shuffled_df = df.sample(frac=1)
            
            # y is last column
            y = np.array(shuffled_df.iloc[:,-1])
            
            # x is all columns except last one.
            X = np.array(shuffled_df.iloc[:,:-1])
            
            # insert a one for the bias term
            ones = np.ones(len(X))
            X_prime = np.insert(X, 0, ones, axis = 1)
            
            epoch_error = 0.0
            
            # for each training example in df
            for j in range(X_prime.shape[0]):
                curr_y = y[j]
                curr_x = X_prime[j]
                
                output = self.__runNetwork(curr_x)
                
                # sum the error to ensure we'
                epoch_error += (curr_y - output)**2
                
                # back propogate and then update weight
                self.__backPropogate(curr_y)
                self.__updateWeights(curr_x)
                
            if self.print_error != 0:
                if i % self.print_error == 0:
                    print('Epoch = %d, error =%.3f' % (i, epoch_error))
        
    def __constructNetwork(self, df):
        """
        Sets up the network framework with normally distributed random weights

        Parameters
        ----------
        df : pandas.dataframe
            dataframe that is used to get shape of input layer.

        Returns
        -------
        None.

        """
        num_inputs = df.shape[1]
        
        network = list()
        
        if not self.all_zeros:
            layer_1 = [{'weights':[np.random.normal() for inputs in range(num_inputs)]} 
                        for node in range(self.width)]
            
            
            
            layer_2 = [{'weights':[np.random.normal() for inputs in range(num_inputs)]}
                        for node in range(self.width)]
            
            
            
            layer_out = [{'weights':[np.random.normal() for inputs in range(self.width)]}]
            
        else:
            layer_1 = [{'weights':[0 for inputs in range(num_inputs)]} 
                        for node in range(self.width)]
            
            
            
            layer_2 = [{'weights':[0 for inputs in range(num_inputs)]}
                        for node in range(self.width)]
            
            
            
            layer_out = [{'weights':[0 for inputs in range(self.width)]}]
            
        
            
        network.append(layer_1)
        network.append(layer_2)
        network.append(layer_out)
        
        self.network = network
        
    def __backPropogate(self, true_y):
        """
        Calculates the error for each neuron and caches it

        Parameters
        ----------
        true_y : int
            The true value of the output.

        Returns
        -------
        None.

        """
        # go backwards through all layers
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            
            # if the current layer is the output layer
            if i == len(self.network)-1:
                neuron = layer[0]
                errors.append(true_y - neuron['output'])
            else:
                # go through all neurons on this layer
                for neuron in range(len(layer)):
                    error = 0.0
                    # get the error of all neurons dependent on this neuronj
                    for previous_neuron in self.network[i+1]:
                        error += previous_neuron['weights'][neuron] * previous_neuron['delta']
                    errors.append(error)
            
            # now find and cache the error for other neurons to use    
            for j in range(len(layer)):
                neuron = layer[j]
                output = neuron['output']
                neuron['delta'] = errors[j] * output * (1.0-output)
                
    def __updateWeights(self, row):
        """
        Updates all of the weights based on the errors

        Parameters
        ----------
        row : numpy.array
            The inputs for the training example.

        Returns
        -------
        None.

        """
        # go through each layer 
        
        learning_rate = self.__getAndUpdateLearningRate()
        for layer in range(len(self.network)):
            inputs = row
            
            # update inputs after first layer
            if layer != 0:
                inputs = [neuron['output'] for neuron in self.network[layer-1]]
                
            for neuron in self.network[layer]:
                for index in range(len(inputs)):
                    neuron['weights'][index] += learning_rate * neuron['delta']
                    
                    
    def __getAndUpdateLearningRate(self):
        """
        Calculates the learning rate for this update and updates t 
        which controls the decay of the growth rate

        Returns
        -------
        rate : double
            The current learning rate.

        """
        rate = self.gamma_0 / (1 + (self.gamma_0 / self.d) * self.t)
        self.t += 1
        return rate
        
            
        
    def __calculateNeuronActivation(self, weights, inputs):
        """
        Calculates the activation of a neuron using the weights, input, and 
        a sigmoidal curve

        Parameters
        ----------
        weights : numpy.array
            array of weights.
        inputs : numpy.array
            array of the inputs.

        Returns
        -------
        double
            The activation of the neuron from 0 to 1.

        """
        activation = 0
        
        for i in range(len(weights)):
            activation += weights[i] * inputs[i]
            
        return expit(activation)
    
    
    
    def __runNetwork(self, row):
        """
        Forward propogates to run network

        Parameters
        ----------
        row : numpy.array
            First layer input.

        Returns
        -------
        input_prime : double
            The output of the network.

        """
        x_in = np.array(row[0:-1])
        input_prime = np.insert(x_in, 0, 1)
        
        # Go through each layer of the network
        for layer in self.network:
            new_input = []
            
            # For each neuron in the layer
            for neuron in layer:
                # find the activation by using the weights and all input from previous layer
                activation = self.__calculateNeuronActivation(neuron['weights'], input_prime)
                # store the output
                neuron['output'] = activation
                
                # the output from this layer is input to next layer
                new_input.append(neuron['output'])
                
            input_prime = new_input
            
        # output of last layer is answer
        return input_prime
                
    
    def Predict(self, row):

        # input is  assumed to have output on it
         x_in = np.array(row[0:-1])
         x_in_prime = np.insert(x_in, 0, 1)
         
         prediction = self.__runNetwork(x_in_prime)[0]
         
         #print(prediction)
         if prediction < 0.5:
             return 0
         else:
             return 1
    
        
    