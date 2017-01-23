from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold
from simplefunctions import *


def runbaggingtree(data, target):
    folds = [10]
    depths = [10, 100, 1000]
    estimators = [100, 1000]
    for fold in folds:

        print('fold = %d ' % fold)

        for depth in depths:
            for estimator in estimators:
                matrices1 = []
                matrices2 = []
                print('depth = %d ' % depth)
                print('estimators = %d ' % estimator)
                clf = tree.DecisionTreeClassifier(max_depth=depth)
                skf = StratifiedKFold(n_splits=fold, random_state=5)
                bagging = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=depth), max_samples=0.5,
                                            max_features=1.0, n_estimators=estimator)
                testpredict, testtarget = cross_val_pred2ict(bagging, data, target, cv=skf, n_jobs=-1)

                if len(testpredict) != len(testtarget):
                    raise ValueError('length score and target are different!')
                for pr, tar in zip(testpredict, testtarget):
                    matrices1.append(confusion_matrix(tar, pr))
                precision1st = precision(matrices1)
                sesnivitivity1st = sensitivity(matrices1)
                print("Accuracy: %r" % str(accuracy(matrices1)))
                print("Precision: %r" % str(precision1st))
                print("Recall: %r" % str(sesnivitivity1st))
                print("f1")
                print(f1tpfp(matrices1))
                print(f1prre(precision1st, sesnivitivity1st))
                print(f1avg(matrices1))

                for matrix in matrices1:
                    matrices2.append(np.array([[matrix[1, 1], matrix[1, 0]], [matrix[0, 1], matrix[0, 0]]]))

                precision2st = precision(matrices2)
                sesnivitivity2st = sensitivity(matrices2)
                print("Accuracy: %r" % str(accuracy(matrices2)))
                print("Precision: %r" % str(precision2st))
                print("Recall: %r" % str(sesnivitivity2st))
                print("f1")
                print(f1tpfp(matrices2))
                print(f1prre(precision2st, sesnivitivity2st))
                print(f1avg(matrices2))
