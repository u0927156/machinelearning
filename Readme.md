This is a machine learning library developed by Spencer Peterson for CS5350/6350 in University of Utah.

Decision Tree:
To use the decision tree, first build a decision tree object, this takes no parameters. 
Then the tree needs to be populated, there are two options for doing this. You may either use 
BuildFromFile or BuildFromDataFrame. 

BuildFromFile takes the filename of a csv, an integer for the maximum depth of the tree, 
an integer to select the method for splitting columns, 0 for entropy, 1 for majority error, and 2 for gini, and an optional
input that is a boolean, if it is true values marked as a string 'unknown' will be converted to the most common value of the column.

BuildFromDataFrame takes a pandas data frame, an integer for maximum depth, an integer to select the method for splitting columns, 
0 for entropy, 1 for majority error, and 2 for gini, a vector of weights with the same number elements as rows of the dataframe, if None is
used for the weights, the tree will behave as if all are weighted the same, and a boolean that  if it is true values that are
a string 'unknown' will be converted to the most common value of the column.

To make a prediction call Predict() with a row of data. 

To check the accuracy, call GetAccuracyLevel with a filename of csv that has test data and the tree will return a decimal that is the
accuracy level of the tree.

Perceptrons:

There are three perceptron classes availabe for use. A standard perceptron, a voted perceptron, and an averaged perceptron.

They all operate in the same way. Create an instance of the class. Then call build perceptron using a dataframe. After that you can make predictions by using a row of test data. 
The perceptron assumes you have left the expected output on because it knows you're lazy. It is very good at perceiving after all