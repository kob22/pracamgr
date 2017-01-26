from sklearn import tree
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

def runtree(data, target):
    folds = [10]
    depths = [10]
    print("------------ TREE ------------")
    lb = preprocessing.LabelEncoder()
    lb.fit(target)
    target1 = lb.transform(target)
    for fold in folds:
        print('fold = %d ' % fold)
        for depth in depths:
            print('depth = %d ' % depth)
            clf = tree.DecisionTreeClassifier()
            skf = StratifiedKFold(n_splits=fold, random_state=5)

            testpredict, testtarget = cross_val_pred2ict(clf, data, target1, cv=skf.get_n_splits(data, target1),
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)
            testpredict = cross_val_predict(clf, data, target1, cv=skf.get_n_splits(data, target1),
                                            n_jobs=-1, method='predict_proba')
            print(roc_auc_score(y_true=target1, y_score=testpredict[:, 1]))
