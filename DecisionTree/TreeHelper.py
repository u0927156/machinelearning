# -*- coding: utf-8 -*-
"""
Helper Methods for DecisionTree.py
Created on Mon Feb  8 11:02:50 2021

@author: Spencer Peterson
"""
import numpy as np

def getMostCommonLabel(df):
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
        return df[df.columns[-1]].mode().iloc[0]
    
def findBestSplit(df, splitMethod, weights):
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
        return getMajorityErrorSplit(df, weights)
    elif splitMethod == 2:
        return getGiniIndexSplit(df, weights)
    else:
        return getEntropySplit(df, weights)
    
    
def getMajorityErrorSplit(df, weights):
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
    return np.argmin(getTotalMajorityError(df, weights) - np.array(getMajorityErrorOfColumns(df, weights)))

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

def getMajorityErrorOfColumns(df, weights):
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
    return MEs


def getGiniIndexSplit(df, weights):
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
    return np.argmin(getTotalGiniIndex(df, weights) - np.array(getGiniOfColumns(df, weights)))

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

def getGiniOfColumns(df, weights):
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
                
                
    return ginis


def getEntropySplit(df, weights):
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
    return np.argmin(getTotalEntropy(df, weights) - getEntropyOfColumns(df, weights))

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

def getEntropyOfColumns(df, weights):
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