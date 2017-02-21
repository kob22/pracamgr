from data import importdata
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document, Section
from pylatex.utils import bold
from classifiers.clf_expert import ensembel_rating
from classifiers.clf_expertCV import ensembel_rating_cv

import os

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['seeds', 'new_thyroid', 'vehicle', 'ionosphere', 'vertebal', 'yeastME3', 'ecoli', 'bupa',
           'horse_colic',
           'german', 'breast_cancer', 'cmc', 'hepatitis', 'haberman', 'transfusion',
           'car', 'glass', 'abalone16_29', 'solar_flare', 'heart_cleveland', 'balance_scale', 'postoperative']

sections = ["Accuracy", "Sensitivity", "Specificity", "F-1 klasa mniejszosciowa", 'G-mean']
random_state = 5
iterations = 10

tables = []
for tab in range(4):
    table = Tabular('c|cccccc')
    table.add_hline()
    table.add_row(('', "CLFE", "CLFE CV", "CLFE F1", "CLFE F1 CV", "CLFE G", "CLFE G CV"))
    table.add_hline()
    tables.append(table)

clf1 = KNeighborsClassifier()
clf2 = tree.DecisionTreeClassifier(max_depth=20)
clf3 = GaussianNB()

prec_clf1 = ensembel_rating(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)])
prec_clf2 = ensembel_rating_cv(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)])

f1_clf1 = ensembel_rating(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)], function_compare='f1tpfp')
f1_clf2 = ensembel_rating_cv(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)], function_compare='f1tpfp')

g_mean_clf1 = ensembel_rating(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)], function_compare='g_meantpfp')
g_mean_clf2 = ensembel_rating_cv(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)],
                                 function_compare='g_meantpfp')

clfs = [prec_clf1, prec_clf2, f1_clf1, f1_clf2, g_mean_clf1, g_mean_clf2]

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
        for iteration in range(iterations):
            clf_ = clone(clf)
            testpredict, testtarget = cross_val_pred2ict(clf_, db.data, db.target, cv=folds, n_jobs=-1)
            scores.append(accsespf1g(testpredict, testtarget))
            print(str(clf))
            print_scores(testpredict, testtarget)

        avgscores = avgaccsespf1g(scores)
        to_decimal = print_to_latex_two_decimal(avgscores)

        for i, score in enumerate(to_decimal):
            rows[i].append(score)
    for table, row in zip(tables, rows):
        max_v = max(row[1:])
        new_row = []
        for item in row:
            if item == max_v:
                new_row.append(bold(max_v))
            else:
                new_row.append(item)

        table.add_row(new_row)

doc = Document("test_expert_clf")
for i, (tab, sec) in enumerate(zip(tables, sections)):
    section = Section(sec)
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
