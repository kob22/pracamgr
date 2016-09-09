from classifiers import svm
from classifiers import tree
from classifiers import adaboost
from classifiers import randomforest
from classifiers import bagging
from data import importdata
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from numpy import array_str
#from sklearn import tree
from sklearn.metrics import confusion_matrix
#german = importdata.importfile("files/german.data-numeric")
german = importdata.importfile("files/bupa.data")

#clf = tree.DecisionTreeClassifier()
#a = clf.fit(german.data,german.target).predict(german.data)
print(german.data)
print(german.target)
#print (confusion_matrix(german.target, a))
tree.runtree(german.data,german.target)
#svm.runsvcn(german.data,german.target)
#randomforest.runforest(german.data,german.target)
#bagging.runbagging(german.data,german.target)
#adaboost.runada(german.data,german.target)