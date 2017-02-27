from data import importdata
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document, Section
from pylatex.utils import bold
from classifiers.meta_clfold import meta_classifier
from classifiers.clf_expert import clf_expert
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

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
    table.add_hline()
    table.add_row(('', "BKNN", "BTREE", "BNB", "ATREE", "ANB", "ESR", "META"))
    table.add_hline()
    tables.append(table)

clf1 = KNeighborsClassifier()
clf2 = tree.DecisionTreeClassifier()
clf3 = GaussianNB()
iterations = 10
prec_clf1 = clf_expert(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)])
m_clf = meta_classifier(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3), ],
                        estimators_ada=[tree.DecisionTreeClassifier(max_depth=3), GaussianNB()],
                        estimators_bag=[tree.DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier()],
                        function_compare='g_meantpfp')

clfs = [BaggingClassifier(KNeighborsClassifier(), n_estimators=100, max_samples=0.9),
        BaggingClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=100, max_samples=0.9),
        BaggingClassifier(GaussianNB(), n_estimators=100, max_samples=0.9)]
clfs.extend([AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=100),
             AdaBoostClassifier(GaussianNB(), n_estimators=100)])

clfs.extend([prec_clf1, m_clf])

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

doc = Document("test_meta_clf")
for i, (tab, sec) in enumerate(zip(tables, sections)):
    section = Section(sec)
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
