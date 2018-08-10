from data import importdata
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import LongTable, Document, Section
from pylatex.utils import bold
from pylatex.basic import TextColor
from pylatex import MultiRow

import os

path = os.path.dirname(os.path.abspath(__file__))

# zbiory danych
dataset = ['seeds', 'new_thyroid', 'vehicle', 'ionosphere', 'vertebal', 'yeastME3', 'ecoli', 'bupa',
           'horse_colic',
           'german', 'breast_cancer', 'cmc', 'hepatitis', 'haberman', 'transfusion',
           'car', 'glass', 'abalone16_29', 'solar_flare', 'heart_cleveland', 'balance_scale', 'postoperative']

sections = ["Accuracy", "Sensitivity", "Specificity", "F-1 klasa mniejszosciowa", 'G-mean']
random_state = 5
tables = []

# liczba iteracji
iterations = 10

# ustawienia baggingu
samp = 0.8
feat = 0.9

# liczbya fold w sprawdzianie krzyzowym
folds = 10

# glebokosc drzew
depths = [None, 3, 5, 7, 10, 15, 20]
estimators = [5, 10, 20, 50]
estimators_name = ['-']
estimators_name.extend(estimators)

for tab in range(5):
    table = LongTable('c|c|ccccccc')
    table.add_hline()
    table.add_row(('Glebokosc drzewa', 'Liczba est.', "-", "3", "5", "7", "10", "15", "20"))
    table.add_hline()
    tables.append(table)
clfs = []
temp_clf = []
for depth in depths:
    temp_clf.append(tree.DecisionTreeClassifier(max_depth=depth))
clfs.append(temp_clf)

# tworzenie klasyfikatorow
for estimator in estimators:
    temp2_clf = []
    for depth in depths:
        temp2_clf.append(
            BaggingClassifier(tree.DecisionTreeClassifier(max_depth=depth), n_estimators=estimator, max_samples=samp,
                              max_features=feat))
    clfs.append(temp2_clf)

for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)

    # obliczanie wynikow
    for id, (clfs_, name) in enumerate(zip(clfs, estimators_name)):
        rows = []
        if id == 0:
            col = MultiRow(5, data=data)
        else:
            col = ''
        for i in range(5):
            rows.append([col, name])
        for clf in clfs_:
            scores = []
            for iteration in range(iterations):
                clf_ = clone(clf)
                # sprawdzian krzyzowy
                testpredict, testtarget = cross_val_pred2ict(clf_, db.data, db.target, cv=folds, n_jobs=-1)
                scores.append(accsespf1g(testpredict, testtarget))
                print(str(clf))
                print_scores(testpredict, testtarget)
            # usrednianie wynikow
            avgscores = avgaccsespf1g(scores)
            to_decimal = print_to_latex_two_decimal(avgscores)
            for i, score in enumerate(to_decimal):
                rows[i].append(score)

        # zapis do tabeli
        for table, row in zip(tables, rows):

            max_v = max(row[2:])
            new_row = []

            for item in row:
                if item == max_v and item > 0.01:
                    new_row.append(bold(max_v))
                else:
                    new_row.append(item)
            table.add_row(new_row)
            if id == 4:
                table.add_hline()
            else:
                table.add_hline(start=2)

doc = Document("bagging_tree_%s%s" % (feat, samp))
for i, tab, in enumerate(tables):
    section = Section(sections[i])
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
