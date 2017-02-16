from data import importdata
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from classifiers.stackingcv import StackingCVClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document, Section
from pylatex.utils import bold
from pylatex.basic import TextColor

import os

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['abalone16_29', 'balance_scale', 'breast_cancer', 'car', 'cmc',
           'ecoli', 'glass', 'haberman', 'heart_cleveland', 'hepatitis',
           'new_thyroid', 'postoperative', 'solar_flare', 'transfusion', 'vehicle',
           'yeastME3', 'bupa', 'german', 'horse_colic', 'ionosphere', 'seeds', 'vertebal']

fold = 10

random_state = 5
estimatorss = [5, 10, 20, 50, 100, 200]
for estimators in estimatorss:
    tables = []
    for tab in range(4):
        table = Tabular('c|cccccccccc')
        table.add_hline()
        table.add_row(('', "Tree", "BG", "(3)", "B(3)", "(5)", "B(5)", "(10)", "B(10)", "(20)", "B(20)"))
        table.add_hline()
        tables.append(table)

    clf1 = tree.DecisionTreeClassifier()
    clf2 = tree.DecisionTreeClassifier(max_depth=3)
    clf3 = tree.DecisionTreeClassifier(max_depth=7)
    clf4 = tree.DecisionTreeClassifier(max_depth=10)
    clf5 = tree.DecisionTreeClassifier(max_depth=15)
    clfs = [clf1, AdaBoostClassifier(tree.DecisionTreeClassifier(), n_estimators=estimators), clf2,
            AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=estimators), clf3,
            AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=7), n_estimators=estimators),
            clf4, AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=10), n_estimators=estimators), clf5,
            AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=15), n_estimators=estimators)]

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
            new_row = [row[0]]

            for id in range(1, len(row[1:]), 2):
                if row[id] == max_v:
                    new_row.append(TextColor('red', row[id]))
                    new_row.append(row[id + 1])
                elif row[id + 1] == max_v:
                    new_row.append(row[id])
                    new_row.append(TextColor('red', row[id + 1]))
                else:
                    if row[id] > row[id + 1]:
                        new_row.append(bold(row[id]))
                        new_row.append(row[id + 1])
                    elif row[id] < row[id + 1]:
                        new_row.append(row[id])
                        new_row.append(bold(row[id + 1]))
                    else:
                        new_row.append(row[id])
                        new_row.append(row[id + 1])

            table.add_row(new_row)
            table.add_hline()

    doc = Document("ada_boost_treee%s" % estimators)
    for i, tab, in enumerate(tables):
        section = Section(str(i))
        section.append(tab)
        doc.append(section)
    doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
    doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))