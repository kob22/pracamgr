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
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
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
    table = Tabular('c|ccccc')
    table.add_row(('', "SMOTE", "ADASYN", "NCR", "SMOTEENN", "SMOTETomek"))
    table.add_hline()
    tables.append(table)
random_st = 5
clf1 = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=50)

methods = [SMOTE(random_state=random_st), ADASYN(random_state=random_st),
           NeighbourhoodCleaningRule(random_state=random_st), SMOTEENN(random_state=random_st),
           SMOTETomek(random_state=random_st)]
names_m = ["SMOTE", "ADASYN", "NCR", "SMOTEENN", "SMOTETomek"]
iterations = 2
random_st = 5

for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)
    length_data = len(data)

    rows = []
    for i in range(5):
        rows.append([data])

    if length_data > 1000:
        folds = 10
    elif length_data > 700:
        folds = 7
    elif length_data > 500:
        folds = 5
    else:
        folds = 3
    skf = StratifiedKFold(n_splits=folds, random_state=random_st)

    for method, name in zip(methods, names_m):

        met = method

        scores = []

        # powtorzenie X razy i obliczenie sredniej
        for iteration in range(iterations):
            predict_re = []
            targets_re = []
            for train_index, test_index in skf.split(db.data, db.target):
                data_re, tar_re = met.fit_sample(db.data[train_index], db.target[train_index])
                clf_ = clone(clf1)

                # trenowanie
                clf_.fit(data_re, tar_re)

                # testowanie
                predict_re.append(clf_.predict(db.data[test_index]))
                targets_re.append(db.target[test_index])
            # obliczanie wyniku ze sprawdzianu krzyzowego
            scores.append(accsespf1g(predict_re, targets_re))
            print(name)
            print(str(clf1))
            print_scores(predict_re, targets_re)

        avgscores = avgaccsespf1g(scores)
        to_decimal = print_to_latex_two_decimal(avgscores)

        for i, score in enumerate(to_decimal):
            rows[i].append(score)
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

doc = Document("compare_methods")
for i, tab, in enumerate(tables):
    section = Section(sections[i])
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
