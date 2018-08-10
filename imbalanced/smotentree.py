from imblearn.combine import SMOTEENN
from sklearn import tree
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

## zostawic to, pokaze sie jako blad takiego postepowania

def runtree(data, target):
    lb = preprocessing.LabelEncoder()
    lb.fit(target)
    target1 = lb.transform(target)
    sm = SMOTEENN()
    clf = tree.DecisionTreeClassifier()
    folds = [3]
    depths = [10]
    print("------------ TREE ------------")

    for fold in folds:
        skf = StratifiedKFold(n_splits=fold, random_state=5)
        test_target = []
        test_predict = []
        test_proba = []
        test_proba_target = []
        for train_index, test_index in skf.split(data, target1):
            clf_ = clone(clf)
            X_resampled, y_resampled = sm.fit_sample(data[train_index], target1[train_index])
            clf_.fit(X_resampled, y_resampled)
            test_predict.append(clf_.predict(data[test_index]))
            test_target.append(target1[test_index])
            test_proba_target.extend(target1[test_index])
            test_proba.extend(clf_.predict_proba(data[test_index])[:, 1])

        print_scores(test_predict, test_target)
        print(roc_auc_score(y_true=test_proba_target, y_score=test_proba))
