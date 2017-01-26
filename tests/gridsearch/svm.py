from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def runsvmgrid(data, target):
    mms = MinMaxScaler()
    datamms = mms.fit_transform(data)
    stdsc = StandardScaler()
    datastdsc = stdsc.fit_transform(data)
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    param_grid = [{'C': param_range,
                   'kernel': ['linear']},
                  {'C': param_range,
                   'gamma': param_range,
                   'kernel': ['rbf']}]

    clf = svm.SVC()

    grid_search = GridSearchCV(clf, scoring='accuracy', cv=10, param_grid=param_grid)
    # scores = cross_val_score(grid_search, data, target, scoring='accuracy', cv=10)

    # print(scores)
    grid_search.fit(data, target)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)

    grid_search.fit(datamms, target)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)

    grid_search.fit(datastdsc, target)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
