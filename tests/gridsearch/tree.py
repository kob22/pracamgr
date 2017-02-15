from sklearn import tree

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import BaggingClassifier

import numpy as np
def runtreegrid(data, target):
    max_depth = [None, 3, 5, 7, 10, 20]
    for depth in max_depth:
        print(depth)
        param_grid = [{'max_features': [0.4, 0.6, 0.7, 0.8, 0.9, 1.0], 'max_samples': [0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
                       'n_estimators': [5, 10, 15, 20, 50, 100]}]
        clf = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=depth))
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

        grid_search.fit(data, target)
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
