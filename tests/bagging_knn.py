from data import importdata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document, Section
from pylatex.utils import bold
from pylatex import MultiRow, LongTable

import os

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['breast_cancer', 'cmc', 'hepatitis', 'haberman', 'glass', 'abalone16_29', 'heart_cleveland', 'postoperative']

sections = ["Accuracy", "Sensitivity", "Specificity", "F-1 klasa mniejszosciowa", 'G-mean']
random_state = 5
# ilosc czesci w sprawdzianie krzyzowym
folds = 10
# liczba iteracji
iterations = 10
tables = []

# liczba sasiadow
n_neighbors = [1, 2, 3, 5, 7]
# liczba klasyfikatorow w bagging
estimators = [5, 10, 20, 50]
estimators_name = ['-']
estimators_name.extend(estimators)


clfs = []
temp_clf = []
for neighbors in n_neighbors:
    temp_clf.append(KNeighborsClassifier(n_neighbors=neighbors))
clfs.append(temp_clf)

# dodawanie klasyfikatorow
for estimator in estimators:
    temp2_clf = []
    for neighbors in n_neighbors:
        temp2_clf.append(
            BaggingClassifier(KNeighborsClassifier(n_neighbors=neighbors), n_estimators=estimator, max_features=1.0,
                              max_samples=0.8))
    clfs.append(temp2_clf)

for tab in range(5):
    table = LongTable('c|c|ccccc')
    table.add_hline()
    table.add_row(('Glebokosc drzewa', 'Liczba est.', "1", "2", "3", "5", "7"))
    table.add_hline()
    tables.append(table)


for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)


    # klasyfikacja
    for id, (clfs_, name) in enumerate(zip(clfs, estimators_name)):
        rows = []
        if id == 0:
            col = MultiRow(5, data=data)
        else:
            col = ''
        for i in range(5):
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
# zapis do pliku
doc = Document("bagging_knn")
for i, tab, in enumerate(tables):
    section = Section(sections[i])
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
