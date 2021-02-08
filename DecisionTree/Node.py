# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 06:54:47 2021

@author: Spencer Peterson
"""

class Node:
    """
    This is a class used to represent a Node in a decision tree
    
    ...
    
    Attributes:
        category: A string represnting the category that this node checks in the decision tree
        
        branches: A dictionary for the branches. The keys are the values the category can take,
        and the values are the nodes this node is attached to.
    
    """
    
    
    def __init__(self, name):
        """
        
        Parameters
        ----------
        name : string
            Determines the category of the node.
            
        Returns
        -------
        None.

        """
        self.category = name
        self.branches = {}
        
        
        
    def addBranch(self, value, node):
        """
        Adds a branch to the node

        Parameters
        ----------
        value : string
            The value of category that this branch represents.
        node : Node
            The Node that is connected by the specific value of node.

        Returns
        -------
        None.

        """
        self.branches[value] = node
        
        
    def addBranches(self, values, nodes):
        """
        Adds several branches to the node

        Parameters
        ----------
        values : A list of strings
            The values of the categories that each branch represents.
        nodes : A list of Nodes
            The nodes that each value connects to, used in the same order as values.

        Returns
        -------
        None.

        """
        
        if len(values) != len(nodes):
            raise ValueError('values and nodes must have the same length')
            
        for i in range(0, len(values)):
            self.branches[values[i]] = nodes[i]