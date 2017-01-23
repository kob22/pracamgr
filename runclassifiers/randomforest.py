from sklearn.ensemble import RandomForestClassifier
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold


def runforest(data, target):
    folds = [10]
    estimators = [100, 1000, 10000]
    print("------------ RANDOM FOREST  ------------")

    for fold in folds:

        print('fold = %d ' % fold)
        for estimator in estimators:
            print('estimators = %d ' % estimator)
            clf = RandomForestClassifier(n_estimators=estimator)

            skf = StratifiedKFold(n_splits=fold, random_state=5)

            testpredict, testtarget = cross_val_pred2ict(clf, data, target, cv=skf.get_n_splits(data, target),
                                                         n_jobs=-1)

            print_scores(testpredict, testtarget)
