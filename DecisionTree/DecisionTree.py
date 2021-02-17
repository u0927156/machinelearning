# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 07:28:38 2021

@author: Spencer Peterson
"""
import numpy as np
import pandas as pd
from Node import Node 
import os
import TreeHelper


class DecisionTree:
      
      
    def __init__(self, filename, maxDepth, InformationGainMethod, MakeUnknownCommon=False):
          """
          Initializes a decision tree
    
          Parameters
          ----------
          filename : string
              file path of csv that will be used to build the decision tree.
              The file must be a CSV with the expected output in the last column of each row
          maxDepth : integer
              The maximum maxDepth the tree will go
          InformationGainMethod : integer
              An integer that selects what type of method will be used to determine information gain
              0 is Shannon entropy, 1 is Majority Error, 2 is gini index
          MakeUnknownCommon : bool
              If true will make unknowns the most common label in the column
              
          Returns
          -------
          None.
    
          """
          if InformationGainMethod not in range(0,3):
              raise ValueError("InformationGainMethod must be 0, 1, or 2")
              
          self.MakeUnknownCommon = MakeUnknownCommon
          df = self.__processCSV(filename)
          #print("CSV Processed")
          self.head = self.__buildTree(df, maxDepth, InformationGainMethod)
          
          
      
    def __processCSV(self, filename):
        """
        Process the CSV from a given filename into a dataframe used by the decision tree.
        Will process the data into a form that the decision tree can use

        Parameters
        ----------
        filename : string
            The path to the csv that will be used to create the dataframe

        Returns
        -------
        The data frame used to build the decision tree

        """
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, filename)
        df = pd.read_csv(filename, header=None)
        
        # Check if each column is all numeric
        for colInd in range(0,len(df.columns)):
            # if the column is numeric, split into below and above median as a boolean
            if pd.to_numeric(df[colInd], errors='coerce').notnull().all():
                #print(colInd, "is all numeric")
                df[colInd] = (df[colInd] <= df[colInd].median())

        # Change unknowns to most common value in column                
        if self.MakeUnknownCommon:            
            for colInd in range(0,len(df.columns)):                
                mode = df[(df[colInd] != 'unknown')][colInd].mode()[0] # make sure not to count unknown when finding most common value
                df.loc[df[colInd]=='unknown', colInd] = 'failure'
            
                
        return df;
    
    def __buildTree(self, df, maxDepth, InformationGainMethod):
        """
        Driver method for building the decision tree using the ID3 algorithm

        Parameters
        ----------
        df : pandas.Dataframe
            The dataframe used to build the decision tree

        Returns
        -------
        The head node of the decision tree. 

        """
        #print("Starting ID3")
        return self.__ID3(df, maxDepth, 0, InformationGainMethod, df)
        
    
    def __ID3(self, df, maxDepth, currDepth,InformationGainMethod, untouched_df):
        """
        Runs the ID3 algorithm to make decision tree

        Parameters
        ----------
        df : pandas.DataFrame
            The subset of data used to build the current Depth.
        maxDepth : int
            The maximum depth the tree will make.
        currDepth : int
            The current depth of the node.
        untouched_df : pandas.DataFrame
            A data frame that contains all of the original data

        Returns
        -------
        Node
            The labeled node for the next step of the decision tree.

        """
        #print("On level ", currDepth, "of", maxDepth)
        #print(df)
        if currDepth >= maxDepth or len(df.columns) == 1 or len(df[df.columns[-1]].unique()) <= 1:
            #print("Exiting")
            return Node(TreeHelper.getMostCommonLabel(df))
            
        
        else:
            BestSplitterCol = TreeHelper.findBestSplit(df,InformationGainMethod)
            
            [SplitDFs, labels] = TreeHelper.SplitDataFrameByColumn(df, BestSplitterCol, untouched_df)
            currNode = Node(str(BestSplitterCol))
            for index in range(0, len(labels)):
                nodeToAdd = None
                #print(SplitDFs[index])
                if len(SplitDFs[index]) == 0:
                    nodeToAdd = Node(TreeHelper.getMostCommonLabel(df))
                else:
                    nodeToAdd = self.__ID3(SplitDFs[index], maxDepth, currDepth+1,InformationGainMethod, untouched_df)
                    
                currNode.addBranch(labels[index],nodeToAdd)
                                   
                
            return currNode

    def Predict(self, row):
        """
        Predicts a label based on a row of data

        Parameters
        ----------
        row : dataframe row
            The row of data that will be predicted.

        Returns
        -------
        string
            The predicted label.

        """
        row = list(row)
        return self.__recursivePrediction(row, self.head)
        
    def __recursivePrediction(self, row, currNode):
        """
        The method used to find the decision tree's predictions

        Parameters
        ----------
        row : dataframe row
            The row with data that has not been checked yet.
        currNode : Node
            The current node being examined.

        Returns
        -------
        TYPE
            the predicted label.

        """
        if(len(currNode.branches) == 0):
            return currNode.category
        else:
            nextIndex = int(currNode.category)
            nextNode = currNode.branches[row[nextIndex]]
            del row[nextIndex]
           # print(row)
            return self.__recursivePrediction(row, nextNode)
        
    def GetAccuracyLevel(self, filename):
        """
        Finds the error rate for a decision tree given test data

        Parameters
        ----------
        filename : test
            Filename of csv containing test data.

        Returns
        -------
        Error rate, calculated by finding number of incorrect answers over total predictions.

        """
        df = self.__processCSV(filename)
        count = 0
        incorrect = 0
        for i in range(0, len(df)):
            count +=1
            if self.Predict(df.iloc[i,:]) != df.iloc[i,-1]:
                incorrect+=1
            
        return incorrect/count
            
            
        
        