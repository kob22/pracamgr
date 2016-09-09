import numpy as np
from sklearn import svm, tree
from sklearn import cross_validation
from sklearn.cross_validation import KFold

data = np.arange(20000).reshape(1000,20)

with open('german.data-numeric', 'r') as f:
    datafile = np.loadtxt(f)

data = datafile[:,:-1]
target = datafile[:,-1]



