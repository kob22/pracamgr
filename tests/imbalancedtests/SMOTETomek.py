from simplefunctions import *
from data import importdata
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from classifiers.stacking import StackingClassifier
from classifiers.stackingcv import StackingCVClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from imblearn.combine import SMOTETomek
from sklearn.model_selection import StratifiedKFold
from pylatex.utils import bold
from pylatex import Tabular, Document, Section
import os

path = os.path.dirname(os.path.abspath(__file__))

dataset = ['seeds', 'new_thyroid', 'vehicle', 'ionosphere', 'vertebal', 'yeastME3', 'ecoli', 'bupa',
           'horse_colic',
           'german', 'breast_cancer', 'cmc', 'hepatitis', 'haberman', 'transfusion',
           'car', 'glass', 'abalone16_29', 'solar_flare', 'heart_cleveland', 'balance_scale', 'postoperative']

sections = ["Accuracy", "Sensitivity", "Specificity", "F-1 klasa mniejszosciowa", 'G-mean']

tables = []
for tab in range(5):
    table = Tabular('c|cccccc')
    table.add_row(('', "Bag TREE", "Bag TREE SMOTET", "AB TREE", "AB TREE SMOTET", "Stacking", "Stacking SMOTET"))
    table.add_hline()
    tables.append(table)

clf1 = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=50)
clf2 = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=50)

meta = MLPClassifier(solver='lbfgs', random_state=1)
stacking = StackingCVClassifier(
    classifiers=[KNeighborsClassifier(), tree.DecisionTreeClassifier(max_depth=3), GaussianNB()],
    meta_classifier=meta)

# liczba powtorzen klasyfikacji
iterations = 10

# liczba fold w sprawdzianie krzyzowym
folds = 10
random_st = 5
clfs = [clf1, clf2, stacking]
for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)
    length_data = len(data)

    rows = []
    for i in range(5):
        rows.append([data])


    skf = StratifiedKFold(n_splits=folds, random_state=random_st)

    for clf in clfs:
        scores = []
        for iteration in range(iterations):
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

        # smotetomek
        smotetomek_ = SMOTETomek(random_state=random_st)

        scores = []

        # powtorzenie X razy i obliczenie sredniej
        for iteration in range(iterations):
            predict_re = []
            targets_re = []
            for train_index, test_index in skf.split(db.data, db.target):
                data_re, tar_re = smotetomek_.fit_sample(db.data[train_index], db.target[train_index])
                clf_ = clone(clf)

                # trenowanie
                clf_.fit(data_re, tar_re)

                # testowanie
                predict_re.append(clf_.predict(db.data[test_index]))
                targets_re.append(db.target[test_index])
            # obliczanie wyniku ze sprawdzianu krzyzowego
            scores.append(accsespf1g(predict_re, targets_re))
            print("SMOTETomek")
            print(str(clf))
            print_scores(predict_re, targets_re)

        avgscores = avgaccsespf1g(scores)
        to_decimal = print_to_latex_two_decimal(avgscores)

        for i, score in enumerate(to_decimal):
            rows[i].append(score)
    for table, row in zip(tables, rows):
        max_v = max(row[1:])
        new_row = []
        new_row.append(row[0])
        row_temp = row[1:]
        id = 1
        for item in row_temp[::2]:
            if item > row_temp[id]:
                new_row.append(bold(item))
                new_row.append(row_temp[id])
            elif item < row_temp[id]:
                new_row.append(item)
                new_row.append(bold(row_temp[id]))
            else:
                new_row.append(item)
                new_row.append(row_temp[id])
            id += 2
        table.add_row(new_row)

doc = Document("overundersampling_SMOTETomek")
for i, tab, in enumerate(tables):
    section = Section(sections[i])
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
