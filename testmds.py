print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image
import pydotplus

# Parameters
n_classes = 3

plot_step = 0.02

# Load data
iris = load_iris()

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)

print(iris.feature_names)
a = ['sepal dlugosc (cm)', 'sepal szerokosc (cm)', 'petal dlugosc (cm)', 'petal szerokosc (cm)']
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=a,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
