from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from data import importdata

import numpy as np

dataset = ['abalone16_29', 'balance_scale', 'breast_cancer', 'car', 'cmc',
           'ecoli', 'glass', 'haberman', 'heart_cleveland', 'hepatitis',
           'new_thyroid', 'postoperative', 'solar_flare', 'transfusion', 'vehicle',
           'yeastME3', 'bupa', 'german', 'horse_colic', 'ionosphere', 'seeds', 'vertebal']
folds = 10
n_estimators = [5, 10, 15, 20, 50, 100, 200]
for estimator in n_estimators:
    print("Liczba klasyfikatorow: %s" % estimator)
    for data in dataset:
        db = getattr(importdata, 'load_' + data)()
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print('Zbior danych: %s' % data)
        importdata.print_info(db.target)
        param_grid = [{'max_features': [0.4, 0.6, 0.7, 0.8, 0.9, 1.0], 'max_samples': [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]}]
        clf = BaggingClassifier(GaussianNB(), n_estimators=estimator)

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
