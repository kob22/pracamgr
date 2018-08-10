from sklearn import tree

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from data import importdata
import numpy as np

folds = 10
dataset = ['abalone16_29', 'balance_scale', 'breast_cancer', 'car', 'cmc',
           'ecoli', 'glass', 'haberman', 'heart_cleveland', 'hepatitis',
           'new_thyroid', 'postoperative', 'solar_flare', 'transfusion', 'vehicle',
           'yeastME3', 'bupa', 'german', 'horse_colic', 'ionosphere', 'seeds', 'vertebal']
folds = 10
for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)
    max_depth = [None, 1, 2, 3, 5, 7, 10, 20]
    for depth in max_depth:
        print(depth)
        param_grid = [{'n_estimators': [5, 10, 15, 20, 30, 50, 80, 100, 200]}]
        clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=depth))

        grid_search = GridSearchCV(clf, scoring='f1', cv=folds, param_grid=param_grid)

        grid_search.fit(db.data, db.target)
        results = grid_search.cv_results_
        best_parameters2 = []

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        if len((np.flatnonzero(results["rank_test_score"] == 2))) > 0:
            best_index2 = np.flatnonzero(results["rank_test_score"] == 2)[0]
            best_parameters2 = results["params"][best_index2]

        best_parameters = results["params"][best_index]
        print(grid_search.best_score_)
        print(best_parameters)
        print(best_parameters2)
