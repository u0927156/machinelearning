# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 07:28:38 2021

@author: Spencer Peterson
"""
import numpy as np
import pandas as pd
import Node 


class DecisionTree:
      
      
    def __init__(self, filename):
          """
          
    
          Parameters
          ----------
          filename : string
              file path of csv that will be used to build the decision tree.
              The file must be a CSV with the expected output in the last column of each row
    
          Returns
          -------
          None.
    
          """
          df = self.__processCSV(filename)
          
          self.head = self.__buildTree(df)
          
          
      
    def __processCSV(self, filename):
        """
        Process the CSV from a given filename into a dataframe used by the decision tree.
        Will process the data into a form that the 

        Parameters
        ----------
        filename : string
            The path to the csv that will be used to create the dataframe

        Returns
        -------
        The data frame used to build the decision tree

        """
      
        return 0;
    
    def __buildTree(self, df):
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
        
        
        return Node("string")
    
    def __ID3(df):
        