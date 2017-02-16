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
dataset = ['abalone0_4']

fold = 10

random_state = 5
tables = []
for tab in range(4):
    table = Tabular('ccccccc')
    table.add_hline()
    table.add_row(('', "C", "L", "F", "B", "A", "G"))
    table.add_hline()
    tables.append(table)
estimators = 100
clf1 = KNeighborsClassifier()
clf2 = tree.DecisionTreeClassifier()
clf3 = GaussianNB()
clfs = [clf1, BaggingClassifier(KNeighborsClassifier(), n_estimators=estimators), clf2,
        BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators=estimators), clf3,
        BaggingClassifier(GaussianNB(), n_estimators=estimators)]

for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)
    rows = []
    for i in range(4):
        rows.append([])

    for clf in clfs:
        clf_ = clone(clf)
        testpredict, testtarget = cross_val_pred2ict(clf_, db.data, db.target, cv=fold, n_jobs=-1)
        scores = print_to_latex_sespf1g(testpredict, testtarget)

        for i, score in enumerate(scores):
            rows[i].append(score)
        print("----------")
        print(str(clf))
        print_scores(testpredict, testtarget)
    for table, row in zip(tables, rows):
        print(row)
        max_v = max(row)
        new_row = [data]
        for item in row:
            if item == max_v:
                new_row.append(bold(max_v))
            else:
                new_row.append(item)

        print(new_row)
        table.add_row(new_row)
        table.add_hline()

doc = Document("bagging")
for i, tab, in enumerate(tables):
    section = Section(str(i))
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
