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
import os

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['abalone0_4', 'abalone0_4_16_29', 'abalone16_29', 'balance_scale', 'breast_cancer',, 'car', 'cmc',
          'ecoli', 'glass', 'haberman', 'heart_cleveland', 'hepatitis',
          'new_thyroid', 'postoperative', 'solar_flare', 'transfusion', 'vehicle', 'yeastME1',
           'yeastME2', 'yeastME3']
'bupa', 'german', 'horse_colic', 'ionosphere', 'seeds', 'vertebal',
fold = 10

random_state = 5
tables = []
for tab in range(4):
    table = Tabular('|c|c|c|c|c|c|c|c|c|')
    table.add_hline()
    table.add_row(('', "Drzewo", "kNN", "NKB", "SVM", "RForest", "BAGGING", "BOOSTING", "STACKING"))
    table.add_hline()
    tables.append(table)

clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = tree.DecisionTreeClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression(C=10.0)

# skf = StratifiedKFold(n_splits=fold, random_state=2)
eclf1 = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

clfs = [tree.DecisionTreeClassifier(), KNeighborsClassifier(), GaussianNB(), SVC(),
        RandomForestClassifier(n_estimators=50), BaggingClassifier(GaussianNB()),
        AdaBoostClassifier(GaussianNB(), n_estimators=50), eclf1]
for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Klasa: %s' % data)
    importdata.print_info(db.target)
    rows = []
    for i in range(4):
        rows.append([data])

    for clf in clfs:
        clf_ = clone(clf)
        testpredict, testtarget = cross_val_pred2ict(clf_, db.data, db.target, cv=fold,
                                                     n_jobs=-1)
        scores = print_to_latex(testpredict, testtarget)

        for i, score in enumerate(scores):
            rows[i].append(score)
        print("----------")
        print(str(clf))
        print_scores(testpredict, testtarget)
    for table, row in zip(tables, rows):
        print(row)
        table.add_row(row)
        table.add_hline()

doc = Document("Doc")
for i, tab, in enumerate(tables):
    section = Section(str(i))
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
