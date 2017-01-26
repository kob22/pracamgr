from sklearn.naive_bayes import GaussianNB
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold


def runNB(data, target):
    folds = [10]
    depths = [10]
    print("------------ NB ------------")

    for fold in folds:
        print('fold = %d ' % fold)
        for depth in depths:
            print('depth = %d ' % depth)
            clf = GaussianNB()
            skf = StratifiedKFold(n_splits=fold, random_state=5)

            testpredict, testtarget = cross_val_pred2ict(clf, data, target, cv=skf.get_n_splits(data, target),
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)
