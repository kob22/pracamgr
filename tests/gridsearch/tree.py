from sklearn import tree

from sklearn.model_selection import GridSearchCV, cross_val_score


def runtreegrid(data, target):
    depths = [x for x in range(1, 11)]
    depths.extend([x for x in xrange(15, 101, 5)])
    depths.append(None)
    print(depths)
    param_grid = [{'max_depth': depths}]
    clf = tree.DecisionTreeClassifier()
    grid_search = GridSearchCV(clf, scoring='accuracy', cv=10, param_grid=param_grid)
    # scores = cross_val_score(grid_search, data, target, scoring='accuracy', cv=10)

    # print(scores)
    grid_search.fit(data, target)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
