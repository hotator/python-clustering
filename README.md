# python-clustering
A simple clustering Class written in python

should work in python 2 and 3

## usage
in main.py:
- generate or import some datapoints to an numpy-array
- choose your clustering approach and instantiate an object with the data from the clustering class
- call the calculate function
- plot the results, every cluster has its own color (only 7 different are included)

## add own clustering algorithms
- create a new Python file in folder cluster (or copy one of the already included ones)
- create a class which inherited from base/Cluster class
- define your calculate function, it should write its result in self.result als list of tuple-lists like 
[[(1,2),(3,4),(5,6)],[(2,3),(5,8)]]
- each sublist contains all points which form a cluster
- that was it :)
- some helper functions are available in the file with this name
enjoy :)
