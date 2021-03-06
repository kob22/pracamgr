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
dataset = ['vehicle', 'ionosphere', 'ecoli',
           'german', 'breast_cancer', 'cmc', 'hepatitis', 'haberman', 'transfusion',
           'car', 'glass', 'abalone16_29', 'heart_cleveland', 'balance_scale', 'postoperative']

sections = ["Accuracy", "Sensitivity", "Specificity", "F-1 klasa mniejszosciowa", 'G-mean']
random_state = 5
tables = []
for tab in range(5):
    table = LongTable('c|c|cccccc')
    table.add_hline()
    table.add_row(('Zbior danych', 'Glebokosc drzewa.', "-", "5", "10", "20", "50", "100"))
    table.add_hline()
    tables.append(table)

# liczba powtorzen klasyfikacji
iterations = 10

# liczba fold w sprawdzianie krzyzowym
folds = 10
#glebokosc drzewa i liczba estymatorow
depths = [None, 1, 3, 5]
estimators = [5, 10, 20, 50, 100]
depths_name = ['-']
depths_name.extend(depths[1:])

clfs = []
temp_clf = []

for depth in depths:
    temp2_clf = [tree.DecisionTreeClassifier(max_depth=depth)]
    for estimator in estimators:
        temp2_clf.append(AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=depth), n_estimators=estimator))
    clfs.append(temp2_clf)

# obliczanie wynikow
for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)


    for id, (clfs_, name) in enumerate(zip(clfs, depths_name)):
        rows = []
        if id == 0:
            col = MultiRow(len(depths), data=data)
        else:
            col = ''
        for i in range(len(depths) + 1):
            rows.append([col, name])
        # obliczenia dla kazdego klasyfikatora
        for clf in clfs_:
            scores = []
            # powtarzanie klasyfikacji
            for iteration in range(iterations):
                clf_ = clone(clf)
                # sprawdzian krzyzowy
                testpredict, testtarget = cross_val_pred2ict(clf_, db.data, db.target, cv=folds, n_jobs=-1)
                scores.append(accsespf1g(testpredict, testtarget))
                print(str(clf))
                print_scores(testpredict, testtarget)
            # usrednanie wynikow
            avgscores = avgaccsespf1g(scores)
            to_decimal = print_to_latex_two_decimal(avgscores)
            for i, score in enumerate(to_decimal):
                rows[i].append(score)

        # dodanie do tabeli
        for table, row in zip(tables, rows):

            max_v = max(row[2:])
            new_row = []
            new_row.extend(row[:2])
            for item in row[2:]:
                if item == max_v and item > 0.01:
                    new_row.append(bold(max_v))
                else:
                    new_row.append(item)

            table.add_row(new_row)
            if id == len(depths_name) - 1:
                table.add_hline()
            else:
                table.add_hline(start=2)

#zapis do pliku
doc = Document("adaboost_tree")
for i, tab, in enumerate(tables):
    section = Section(sections[i])
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
