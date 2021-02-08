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
    
def findBestSplit(df, splitMethod):
    if splitMethod == 1:
        return getMajorityErrorSplit(df)
    elif splitMethod == 2:
        return getGiniIndexSplit(df)
    else:
        return getEntropySplit(df)
    
    
def getMajorityErrorSplit(df):
    return 0

def getGiniIndexSplit(df):
    return 0

def getEntropySplit(df):
    return np.argmin(getTotalEntropy(df) - getEntropyOfColumns(df))

def getTotalEntropy(df):
    entropy_S = 0
    total_examples = len(df.index)
    
    for value in  (df[df.columns[-1]].unique()):
        p = len(df[df.iloc[:,-1]==value])/total_examples
        entropy = -p * np.log2(p)
        entropy_S = entropy_S + entropy
    return entropy_S

def getEntropyOfColumns(df):
    
    # Get the labels
    labels = (df[df.columns[-1]].unique())
    entropies = []
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
                    
                    sum_entropy_label = sum_entropy_label + entropy * p
        
            entropies.append(sum_entropy_label)
    return entropies

def SplitDataFrameByColumn(df, colToSplitBy):
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
    label_values : TYPE
        the values of the labels that column was split by.

    """
    SplitRegions = []
    label_values =(df[df.columns[colToSplitBy]].unique())
    for value in  label_values:
        Sv = df[df.iloc[:,colToSplitBy]==value]
        Sv = Sv.drop(Sv.columns[colToSplitBy], axis=1) 
        SplitRegions.append(Sv)
    return SplitRegions, label_values