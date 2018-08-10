from sklearn import tree

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from data import importdata
import numpy as np

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
    max_depth = [None, 3, 5, 7, 10, 20]
    n_estimators = [5, 10, 15, 20, 50, 100]
    for estimator in n_estimators:
        for depth in max_depth:
            param_grid = [
                {'max_features': [0.4, 0.6, 0.7, 0.8, 0.9, 1.0], 'max_samples': [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]}]
            clf = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=depth), n_estimators=estimator)
            length_data = len(data)
            if length_data > 1000:
                folds = 10
            elif length_data > 700:
                folds = 7
            elif length_data > 500:
                folds = 5
            else:
                folds = 3
            grid_search = GridSearchCV(clf, scoring='f1', cv=folds, param_grid=param_grid)

            grid_search.fit(db.data, db.target)
            results = grid_search.cv_results_
            best_parameters2 = []

            best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
            if len((np.flatnonzero(results["rank_test_score"] == 2))) > 0:
                best_index2 = np.flatnonzero(results["rank_test_score"] == 2)[0]
                best_parameters2 = results["params"][best_index2]

            best_parameters = results["params"][best_index]
            print(best_parameters)
            print(best_parameters2)
