from data import importdata
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import LongTable, Document, Section
from pylatex.utils import bold
from pylatex.basic import TextColor
from pylatex import MultiRow

import os

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['seeds', 'new_thyroid', 'vehicle', 'ionosphere', 'vertebal', 'yeastME3', 'ecoli', 'bupa',
           'horse_colic',
           'german', 'breast_cancer', 'cmc', 'hepatitis', 'haberman', 'transfusion',
           'car', 'glass', 'abalone16_29', 'solar_flare', 'heart_cleveland', 'balance_scale', 'postoperative']

sections = ["Accuracy", "Sensitivity", "Specificity", "F-1 klasa mniejszosciowa", 'G-mean']
random_state = 5
tables = []
for tab in range(5):
    table = LongTable('c|c|ccccccc')
    table.add_hline()
    table.add_row(('Glebokosc drzewa', 'Liczba est.', "-", "3", "5", "7", "10", "15", "20"))
    table.add_hline()
    tables.append(table)
depths = [None, 3, 5, 7, 10, 15, 20]
estimators = [5, 10, 20, 50, 100, 200]
estimators_name = ['-']
estimators_name.extend(estimators)
print(estimators_name)
clfs = []
temp_clf = []
for depth in depths:
    temp_clf.append(tree.DecisionTreeClassifier(max_depth=depth))
clfs.append(temp_clf)

for estimator in estimators:
    temp2_clf = []
    for depth in depths:
        temp2_clf.append(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=depth), n_estimators=estimator))
    clfs.append(temp2_clf)

for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)

    length_data = len(data)
    if length_data > 1000:
        folds = 10
    elif length_data > 700:
        folds = 7
    elif length_data > 500:
        folds = 5
    else:
        folds = 3

    for id, (clfs_, name) in enumerate(zip(clfs, estimators_name)):
        rows = []
        if id == 0:
            col = MultiRow(len(estimators) + 1, data=data)
        else:
            col = ''
        for i in range(len(estimators) + 1):
            rows.append([col, name])
        for clf in clfs_:
            clf_ = clone(clf)
            testpredict, testtarget = cross_val_pred2ict(clf_, db.data, db.target, cv=folds, n_jobs=-1)
            scores = print_to_latex_accsespf1g(testpredict, testtarget)

            for i, score in enumerate(scores):
                rows[i].append(score)
            print("----------")
            print(str(clf))
            print_scores(testpredict, testtarget)
        for table, row in zip(tables, rows):

            max_v = max(row[2:])
            new_row = []

            for item in row:
                if item == max_v and item > 0.01:
                    new_row.append(bold(max_v))
                else:
                    new_row.append(item)
            table.add_row(new_row)
            if id == len(estimators):
                table.add_hline()
            else:
                table.add_hline(start=2)

doc = Document("adaboost_tree_2")
for i, tab, in enumerate(tables):
    section = Section(sections[i])
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
