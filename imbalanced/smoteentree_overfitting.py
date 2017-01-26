from imblearn.combine import SMOTEENN
from sklearn import tree
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


## zostawic to, pokaze sie jako blad takiego postepowania

def runtree(data, target):
    lb = preprocessing.LabelEncoder()
    lb.fit(target)
    target1 = lb.transform(target)
    sm = SMOTEENN()
    X_train, X_test, y_train, y_test = train_test_split(data, target1, test_size=0.40, random_state=5, stratify=target1)
    print(y_test.size)
    print(np.bincount(y_test))
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    folds = [10]
    depths = [10]
    print("------------ TREE ------------")

    for fold in folds:
        print('fold = %d ' % fold)
        for depth in depths:
            print('depth = %d ' % depth)
            clf = tree.DecisionTreeClassifier()
            skf = StratifiedKFold(n_splits=fold, random_state=5)

            testpredict, testtarget = cross_val_pred2ict(clf, data, target1, cv=skf.get_n_splits(data, target1),
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)
            print('smotens')
            testpredict, testtarget = cross_val_pred2ict(clf, X_resampled, y_resampled,
                                                         cv=skf.get_n_splits(X_resampled, y_resampled),
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)

            testpredict = cross_val_predict(clf, data, target1, cv=skf.get_n_splits(data, target1),
                                            n_jobs=-1, method='predict_proba')
            print(roc_auc_score(y_true=target1, y_score=testpredict[:, 1]))

            testpredict = cross_val_predict(clf, X_resampled, y_resampled,
                                            cv=skf.get_n_splits(X_resampled, y_resampled),
                                            n_jobs=-1, method='predict_proba')
            print(roc_auc_score(y_true=y_resampled, y_score=testpredict[:, 1]))

            clf.fit(X_resampled, y_resampled)
            print_scores([clf.predict(X_test)], [y_test])
            print(roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1]))
