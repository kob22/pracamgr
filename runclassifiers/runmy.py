from data import importdata
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document, Section
from pylatex.utils import bold
from sklearn.ensemble import VotingClassifier
from classifiers.ensemble_rating import ensembel_rating
from classifiers.ensemble_ratingcv import ensembel_rating_cv
from sklearn.neural_network import MLPClassifier
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['abalone0_4', 'abalone041629', 'abalone16_29', 'balance_scale', 'breast_cancer', 'car', 'cmc',
           'ecoli', 'glass', 'haberman', 'heart_cleveland', 'hepatitis',
           'new_thyroid', 'postoperative', 'solar_flare', 'transfusion', 'vehicle', 'yeastME1',
           'yeastME2', 'yeastME3', 'bupa', 'german', 'horse_colic', 'ionosphere', 'seeds', 'vertebal']

random_state = 5
tables = []
for tab in range(4):
    table = Tabular('c|cccccc')
    table.add_hline()
    table.add_row(('', "KNN", "TREE", "GB", "MYCLF", "VT", "CLF2"))
    table.add_hline()
    tables.append(table)
estimators = 50
clf1 = KNeighborsClassifier()
clf2 = tree.DecisionTreeClassifier()
clf3 = GaussianNB()
clf4 = MLPClassifier(solver='lbfgs', random_state=1)

clfs = [BaggingClassifier(KNeighborsClassifier, n_estimators=100),
        BaggingClassifier(tree.DecisionTreeClassifier, n_estimators=100),
        BaggingClassifier(GaussianNB, n_estimators=100)]
clfs = [AdaBoostClassifier(tree.DecisionTreeClassifier, n_estimators=100),
        AdaBoostClassifier(GaussianNB, n_estimators=100)]
myclf = ensembel_rating(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)])
voting = VotingClassifier(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)], voting='hard')
myclf2 = ensembel_rating_cv(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)])
