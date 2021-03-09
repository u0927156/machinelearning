# -*- coding: utf-8 -*-
"""
Helper Methods for DecisionTree.py
Created on Mon Feb  8 11:02:50 2021

@author: Spencer Peterson
"""
import numpy as np
import pandas as pd
import random
def getMostCommonLabel(df, weights=None):
        """
        Gets the most common label from a data frame. Labels are considered the last column in a dataframe

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to analyze.

        Returns
        -------
        The most common label as a string.

        """
        if weights is None:
            return df[df.columns[-1]].mode().iloc[0]
        else:
            labels = {}
            for i in range(0,len(weights)):
                label = df[df.columns[-1]].iloc[i]
                if label in labels:
                    labels[label] += weights[i]
                else:
                    labels[label] = weights[i]
                
            labelWeight = -10000
        
        
            LabelToReturn = ''
            for key in labels:
                if labels[key] > labelWeight:
                    labelWeight = labels[key]
                    LabelToReturn = key
                    
            return LabelToReturn
        
        
def findBestSplit(df, splitMethod, weights, RandomAttributes):
    """
    Finds the column that provides the most information about the label

    Parameters
    ----------
    df : pandas.dataframe
        The data frame that is going to be split.
    splitMethod : int
        Selection for split method, 1 is Majority error, 2 is gini, all other numbers are Entropy.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if splitMethod == 1:
        return getMajorityErrorSplit(df, weights, RandomAttributes)
    elif splitMethod == 2:
        return getGiniIndexSplit(df, weights, RandomAttributes)
    else:
        return getEntropySplit(df, weights, RandomAttributes)
    
    
def getMajorityErrorSplit(df, weights, RandomAttributes):
    """
    Gets the column to split using Majority Error

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to split.

    Returns
    -------
    int
        The index of the column that is the best split.

    """
    return np.argmax(getTotalMajorityError(df, weights) - np.array(getMajorityErrorOfColumns(df, weights, RandomAttributes)))

def getTotalMajorityError(df, weights):
    """
    Gets the Majority error for the entire dataset

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to find the Majority Error.

    Returns
    -------
    float
        The majority error of the dataset.

    """
    errors = []
    if weights is None:
        total_examples = len(df.index)
        
        for value in  (df[df.columns[-1]].unique()):
            p = len(df[df.iloc[:,-1]==value])/total_examples
            errors.append(p)
    else:
        total_example_weight = sum(weights)
        for value in  (df[df.columns[-1]].unique()):
            p = sum(weights[df.iloc[:,-1]==value])/total_example_weight
            errors.append(p)
    
        
    
    return 1-max(errors)

def getMajorityErrorOfColumns(df, weights, RandomAttributes):
    """
    Gets the weighted majority error of each column

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe being split.

    Returns
    -------
    MEs : List of floats
        The weighted list of Majority Errors in the same order as the columns
        in df.

    """
    # Get the labels
    labels = (df[df.columns[-1]].unique())
    MEs = []
    
    if weights is None:
        for i in range(0, len(df.columns)-1): 
            if len(df[df.columns[i]].unique()) <= 1:
                MEs.append(0)
            else:
                for value in  (df[df.columns[i]].unique()):
                    MEcol = []
                    Sv = df[df.iloc[:,i]==value]
                    for label in labels:
                        p = len(Sv[Sv.iloc[:,-1]==label])/len(Sv)
                        MEcol.append(p)
            
                MEs.append(1-max(MEcol))
    else:
        for i in range(0, len(df.columns)-1): 
            if len(df[df.columns[i]].unique()) <= 1:
                MEs.append(0)
            else:
                for value in  (df[df.columns[i]].unique()):
                    MEcol = []
                    Sv = df[df.iloc[:,i]==value]
                    Sv_weights = weights[df.iloc[:,i]==value]
                    total_sum_weights = sum(Sv_weights)
                    for label in labels:
                        p = sum(weights[Sv.iloc[:,-1]==label])/total_sum_weights
                        MEcol.append(p)
            
                MEs.append(1-max(MEcol))
    
    if RandomAttributes is not None:
        NumToIgnore = len(MEs) - RandomAttributes
        if NumToIgnore > 0:
            for ind in random.sample(range(len(MEs), k=NumToIgnore)):
                MEs[ind] = 1
        
    return MEs


def getGiniIndexSplit(df, weights, RandomAttributes):
    """
    Get column index of split using Gini Index

    Parameters
    ----------
    df : pandas.DataFrame
        The data being split.

    Returns
    -------
    int
        The column index of the best split.

    """
    #print(getTotalGiniIndex(df), getGiniOfColumns(df),getTotalGiniIndex(df) - np.array(getGiniOfColumns(df)))
    return np.argmax(getTotalGiniIndex(df, weights) - np.array(getGiniOfColumns(df, weights, RandomAttributes)))

def getTotalGiniIndex(df, weights):
    """
    Gets the total gini index for the labels of the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        The data set.

    Returns
    -------
    float
        The gini index of the dataset labels.

    """
    gini_S = 0
    
    if weights is None:
        total_examples = len(df.index)
        
        for label in (df[df.columns[-1]].unique()):
            p = len(df[df.iloc[:,-1]==label])/total_examples
            gini = p**2 # Square the probability
            gini_S += gini
    else:
        total_weight = sum(weights)
        
        for label in (df[df.columns[-1]].unique()):
            p = sum(weights[df.iloc[:,-1]==label])/total_weight
            gini = p**2 # Square the probability
            gini_S += gini
    return 1-gini_S

def getGiniOfColumns(df, weights, RandomAttributes):
    """
    Gets the weighted gini index for each column

    Parameters
    ----------
    df : pandas.DataFrame
        The Dataset.

    Returns
    -------
    ginis : list of floats
        The gini indices of each column as they appear in the dataset.

    """
    # Get the labels
    labels = (df[df.columns[-1]].unique())
    ginis = []
    
    if weights is None:
        for i in range(0, len(df.columns)-1): 
                if len(df[df.columns[i]].unique()) <= 1:
                    ginis.append(0)
                else:
                    for value in  (df[df.columns[i]].unique()):
                        Sv = df[df.iloc[:,i]==value]
                        sum_gini_label = 0
                        for label in labels:
                            p = len(Sv[Sv.iloc[:,-1]==label])/len(Sv)
                            
                            gini = p**2
                            
                            sum_gini_label += gini
                
                    ginis.append((1-sum_gini_label)*len(Sv)/len(df.iloc[:,i]))
                    
    else:       
        for i in range(0, len(df.columns)-1): 
            if len(df[df.columns[i]].unique()) <= 1:
                ginis.append(0)
            else:
                for value in  (df[df.columns[i]].unique()):
                    Sv = df[df.iloc[:,i]==value]
                    Sv_weights = weights[df.iloc[:,i]==value]
                    sum_gini_label = 0
                    for label in labels:
                        p = sum(Sv_weights[Sv.iloc[:,-1]==label])/sum(Sv_weights)
                        
                        gini = p**2
                        
                        sum_gini_label += gini
            
                ginis.append((1-sum_gini_label)*sum(Sv_weights)/sum(weights))    
                
    if RandomAttributes is not None:
        NumToIgnore = len(ginis) - RandomAttributes
        if NumToIgnore > 0:
            for ind in random.sample(range(len(ginis)), k=NumToIgnore):
                ginis[ind] = 1  
              
    return ginis


def getEntropySplit(df, weights, RandomAttributes):
    """
    Finds the best column to split based on entropy

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset.

    Returns
    -------
    int
        The index of the best column to split.

    """
    return np.argmax(getTotalEntropy(df, weights) - getEntropyOfColumns(df, weights, RandomAttributes))

def getTotalEntropy(df, weights):
    """
    Gets the entropy of the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        The data set

    Returns
    -------
    entropy_S : float
        The entropy of the data set.

    """
    entropy_S = 0
    
    
    if weights is None:
        total_examples = len(df.index)
        for value in  (df[df.columns[-1]].unique()):
            p = len(df[df.iloc[:,-1]==value])/total_examples
            entropy = -p * np.log2(p)
            entropy_S = entropy_S + entropy
    else:
        total_weight = sum(weights)
        for value in  (df[df.columns[-1]].unique()):
            p = sum(weights[df.iloc[:,-1]==value])/total_weight
            entropy = -p * np.log2(p)
            entropy_S = entropy_S + entropy
        
        
    return entropy_S

def getEntropyOfColumns(df, weights, RandomAttributes):
    """
    Gets the weighted entropy of each column

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset.

    Returns
    -------
    entropies : list of floats
        The weighted entropy of each column in the order they appear 
        in the dataset.

    """
    # Get the labels
    labels = (df[df.columns[-1]].unique())
    entropies = []
    
    if weights is None:
        for i in range(0, len(df.columns)-1): 
            if len(df[df.columns[i]].unique()) <= 1:
                entropies.append(0)
            else:
                for value in  (df[df.columns[i]].unique()):
                    Sv = df[df.iloc[:,i]==value]
                    sum_entropy_label = 0
                    for label in labels:
                        p = len(Sv[Sv.iloc[:,-1]==label])/len(Sv)
                        
                        if p == 0:
                            entropy = 0
                        else:
                            entropy = -p * np.log2(p)
                        
                        sum_entropy_label = sum_entropy_label + entropy * len(Sv)/len(df.iloc[:,i])
            
                entropies.append(sum_entropy_label)
    else:
        for i in range(0, len(df.columns)-1): 
            if len(df[df.columns[i]].unique()) <= 1:
                entropies.append(0)
            else:
                for value in  (df[df.columns[i]].unique()):
                    Sv = df[df.iloc[:,i]==value]
                    Weights_sv = weights[df.iloc[:,i]==value]
                    
                    sum_entropy_label = 0
                    for label in labels:
                        p = sum(Weights_sv[Sv.iloc[:,-1]==label])/sum(Weights_sv)
                        
                        if p == 0:
                            entropy = 0
                        else:
                            entropy = -p * np.log2(p)
                        
                        sum_entropy_label = sum_entropy_label + entropy * sum(Weights_sv)/sum(weights)
            
                entropies.append(sum_entropy_label)
                  
    # If we have to choose random attributes, just set the entropy to one so it will never be chosen. 
    if RandomAttributes is not None:
        NumToIgnore = len(entropies) - RandomAttributes
        if NumToIgnore > 0:
            for ind in random.sample(range(len(entropies), k=NumToIgnore)):
                entropies[ind] = 1  
            
    return entropies

def SplitDataFrameByColumn(df, colToSplitBy, untouched_df, weights):
    """
    Splits the dataframe by the given column. Returns smaller data frames and 
    the labels they were split by

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be split.
    colToSplitBy : integer
        The column index that the dataframe will be divided by.

    Returns
    -------
    SplitRegions : pandas.DataFrame
        The data frame split into sections based on values on the given column
    label_values : list of strings
        the values of the labels that column was split by.
    Split_Weights : list of numpy arrays
        The weights for each of the split sections

    """
    SplitRegions = []
    Split_Weights = []
    label_values =(untouched_df[untouched_df.columns[df.columns[colToSplitBy]]].unique()) # ensure we get all possible labels
    for value in  label_values:
        Sv = df[df.iloc[:,colToSplitBy]==value] # get the dataset where its equal to label
        Sv = Sv.drop(Sv.columns[colToSplitBy], axis=1) # 
        
        if weights is not None:
            Split_Weights.append(weights[df.iloc[:,colToSplitBy]==value]) # add split weights
        else:
            Split_Weights.append(None)
        SplitRegions.append(Sv)
    return SplitRegions, label_values, Split_Weights

def ProcessDataFrame(df, MakeUnknownCommon=False):
         # Check if each column is all numeric
    for colInd in range(0,len(df.columns)):
        # if the column is numeric, split into below and above median as a boolean

        if pd.to_numeric(df.iloc[:,colInd], errors='coerce').notnull().all():
            #print(colInd, "is all numeric")
            df.iloc[:,colInd] = (df.iloc[:,colInd] <= df.iloc[:,colInd].median())

    # Change unknowns to most common value in column                
    if MakeUnknownCommon:            
        for colInd in range(0,len(df.columns)):                
            mode = df[(df[df.columns[colInd]] != 'unknown')].iloc[:,colInd].mode()[0] # make sure not to count unknown when finding most common value
            df.loc[df[df.columns[colInd]]=='unknown', colInd] = mode
    