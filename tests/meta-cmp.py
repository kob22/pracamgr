from data import importdata
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document, Section
from pylatex.utils import bold
from classifiers.stacking import StackingClassifier
from sklearn.neural_network import MLPClassifier

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
    table = Tabular('c|ccccccc')
    table.add_row(('', "Bag NB", "Bag TREE", "Bag kNN", "AB NB", "AB Tree", "RF", "Stacking"))
    table.add_hline()
    tables.append(table)
clf1 = BaggingClassifier(GaussianNB(), n_estimators=100)
clf2 = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=100)
clf3 = BaggingClassifier(KNeighborsClassifier(), n_estimators=100)
clf4 = AdaBoostClassifier(GaussianNB(), n_estimators=100)
clf5 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=100)
clf6 = RandomForestClassifier(n_estimators=100, )
meta = MLPClassifier(solver='lbfgs', random_state=1)
stacking = StackingClassifier(
    classifiers=[KNeighborsClassifier(), tree.DecisionTreeClassifier(max_depth=3), GaussianNB()],
    meta_classifier=meta)

clfs = [clf1, clf2, clf3, clf4, clf5, clf6, stacking]
for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)
    rows = []
    for i in range(5):
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
        scores = []
        for iteration in range(10):
            clf_ = clone(clf)
            testpredict, testtarget = cross_val_pred2ict(clf_, db.data, db.target, cv=folds, n_jobs=-1)
            scores.append(accsespf1g(testpredict, testtarget))

        avgscores = avgaccsespf1g(scores)
        to_decimal = print_to_latex_two_decimal(avgscores)

        for i, score in enumerate(to_decimal):
            rows[i].append(score)
        print("----------")
        print(str(clf))
        print_scores(testpredict, testtarget)
    for table, row in zip(tables, rows):
        max_v = max(row[1:])
        new_row = []
        for item in row:
            if item == max_v:
                new_row.append(bold(max_v))
            else:
                new_row.append(item)
        table.add_row(new_row)

doc = Document("Meta CMP4")
for i, tab, in enumerate(tables):
    section = Section(sections[i])
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
