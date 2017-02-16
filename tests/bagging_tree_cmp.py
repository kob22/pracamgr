from data import importdata
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from classifiers.stackingcv import StackingCVClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document, Section
from pylatex.utils import bold

import os

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['abalone0_4', 'abalone041629', 'abalone16_29', 'balance_scale', 'breast_cancer', 'car', 'cmc',
           'ecoli', 'glass', 'haberman', 'heart_cleveland', 'hepatitis',
           'new_thyroid', 'postoperative', 'solar_flare', 'transfusion', 'vehicle', 'yeastME1',
           'yeastME2', 'yeastME3', 'bupa', 'german', 'horse_colic', 'ionosphere', 'seeds', 'vertebal']

random_state = 5
tables = []
for tab in range(4):
    table = Tabular('c|cccccccccc')
    table.add_hline()
    table.add_row(('', "Tree", "BG", "(3)", "B(3)", "(5)", "B(5)", "(10)", "B(10)", "(20)", "B(20)"))
    table.add_hline()
    tables.append(table)
estimators = 50
clfs = [BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators=estimators),
        BaggingClassifier(tree.DecisionTreeClassifier(max_depth=2), n_estimators=estimators),
        BaggingClassifier(tree.DecisionTreeClassifier(max_depth=7), n_estimators=estimators),
        clf4, BaggingClassifier(tree.DecisionTreeClassifier(max_depth=15), n_estimators=estimators), clf5,
        BaggingClassifier(tree.DecisionTreeClassifier(max_depth=20), n_estimators=estimators)]

for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)
    rows = []
    for i in range(4):
        rows.append([data])

    length_data = len(data)
    if length_data > 1000:
        folds = 10
    elif length_data > 700:
        folds = 7
    elif length_data > 500:
        folds = 5
    else:
        folds = 3

    for clf in clfs:
        clf_ = clone(clf)
        testpredict, testtarget = cross_val_pred2ict(clf_, db.data, db.target, cv=folds, n_jobs=-1)
        scores = print_to_latex_sespf1g(testpredict, testtarget)

        for i, score in enumerate(scores):
            rows[i].append(score)
        print("----------")
        print(str(clf))
        print_scores(testpredict, testtarget)
    for table, row in zip(tables, rows):
        print(row)
        max_v = max(row[1:])
        new_row = []
        for item in row:
            if item == max_v:
                new_row.append(bold(max_v))
            else:
                new_row.append(item)

        print(new_row)
        table.add_row(new_row)
        table.add_hline()

doc = Document("bagging_tree")
for i, tab, in enumerate(tables):
    section = Section(str(i))
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
