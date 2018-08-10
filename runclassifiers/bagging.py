from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold
from simplefunctions import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def runbaggingtree(data, target):
    folds = [3]
    depths = [5]
    estimators = [50]
    for fold in folds:

        print('fold = %d ' % fold)

        for depth in depths:
            for estimator in estimators:

                print('depth = %d ' % depth)
                print('estimators = %d ' % estimator)

                skf = StratifiedKFold(n_splits=fold, random_state=5)
                bagging = BaggingClassifier(KNeighborsClassifier(), n_estimators=estimator)
                testpredict, testtarget = cross_val_pred2ict(bagging, data, target, cv=10, n_jobs=-1)

                print_scores(testpredict, testtarget)
