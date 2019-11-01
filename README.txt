#
#	Author: Esteban Aranda, Noah Van Stralen
#	Date: 09/08/19
#	Course: CS 4341
#

RUNNING:
Go to the correct directory and run 'python [name_of_file].py'. 

The python files are named according to their content. For example, Task2Part1.py contains
a ANN with no dropout, as described in the project description.

Some files can be ran as they are without any changes, such as 'Task2Part1.py', 'Task2Part2.py',
and 'Task3.py'. As for the other files, there are variables that need to be changed each time 
the file is ran to test the different effects they have on the neural net. For 'Task4Part1.py'
the variable 'hiddenLayers' specifies the number of hidden layers, which for this part could
be 1,2, or 10. For 'Task4Part2.py' the variable 'batchSize' specifies the batch size, which
could be 32 or 512. For 'Task4Experiment1.py' we decided to change the number of nodes
on the first layer, specified by the variable 'nodes', which we tested with 50 and 784.
For 'Task4Experiment2.py' we experimented with the number of folds in the k-fold cross
validation. This can be changed with the variable 'folds', which we tested with 3, 5, and 10.
For 'Task4Experiment3.py' we experimented with dropout on the first layer,
 which can be changed with the variable 'dropOut'. We tested it with 0, 0.2, and 0.5.

