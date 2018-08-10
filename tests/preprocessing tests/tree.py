from data import importdata
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document

dataset = ['abalone0_4', 'abalone0_4_16_29', 'abalone16_29', 'balance_scale', 'breast_cancer', 'bupa', 'car', 'cmc',
           'ecoli', 'german', 'glass', 'haberman', 'heart_cleveland', 'hepatitis', 'horse_colic', 'ionosphere',
           'new_thyroid', 'postoperative', 'seeds', 'solar_flare', 'transfusion', 'vehicle', 'vertebal', 'yeastME1',
           'yeastME2', 'yeastME3']
dataset = ['postoperative']
fold = 10
depths = [1, 2, 3, 5, 7, 10, None]
random_state = 5

table1 = Tabular('|c|c|c|c|c|c|c|c|')
table1.add_hline()
table1.add_row(('', 1, 2, 3, 5, 7, 10, None))
table1.add_hline()
for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print('Klasa: %s' % data)
    importdata.print_info(db.target)
    row = [data]
    for depth in depths:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        testpredict, testtarget = cross_val_pred2ict(clf, db.data, db.target, cv=fold,
                                                     n_jobs=-1)
        row.append(float("{0:.2f}".format(print_to_latex(testpredict, testtarget))))
        print_scores(testpredict, testtarget)
    table1.add_row(row)
    table1.add_hline()

doc = Document("multirow")
doc.append(table1)
doc.generate_tex()
