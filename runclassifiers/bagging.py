from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold
from simplefunctions import *


def runbaggingtree(data, target):
    folds = [3]
    depths = [5]
    estimators = [100, 1000, 10000]
    for fold in folds:

        print('fold = %d ' % fold)

        for depth in depths:
            for estimator in estimators:

                print('depth = %d ' % depth)
                print('estimators = %d ' % estimator)

                skf = StratifiedKFold(n_splits=fold, random_state=5)
                bagging = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=depth), n_estimators=estimator)
                testpredict, testtarget = cross_val_pred2ict(bagging, data, target, cv=skf, n_jobs=-1)

                print_scores(testpredict, testtarget)
