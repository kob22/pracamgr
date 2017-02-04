from sklearn import tree
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
def runtree(data, target):
    folds = [10]
    depths = [10]
    print("------------ TREE ------------")
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        random_state=2)
    for fold in folds:
        print('fold = %d ' % fold)
        for depth in depths:
            print('depth = %d ' % depth)
            clf = tree.DecisionTreeClassifier()
            skf = StratifiedKFold(n_splits=fold, random_state=5)

            testpredict, testtarget = cross_val_pred2ict(clf, data, target, cv=skf.get_n_splits(data, target),
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)
            clf.fit(X_train, y_train)
            print_scores([clf.predict(X_test)], [y_test])
            testpredict = cross_val_predict(clf, data, target, cv=skf.get_n_splits(data, target),
                                            n_jobs=-1, method='predict_proba')
            print(roc_auc_score(y_true=target, y_score=testpredict[:, 1]))

    # Create a pipeline

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Classify and report the results
    a = clf.predict(X_test)
    print(classification_report_imbalanced(y_test, a))
    print_scores([a], [y_test])
