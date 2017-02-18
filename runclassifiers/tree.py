from sklearn import tree
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score

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
            clf1 = tree.DecisionTreeClassifier(max_depth=1)
            clf2 = tree.DecisionTreeClassifier(max_depth=1)
            length_data = len(data)
            if length_data > 1000:
                folds = 10
            elif length_data > 700:
                folds = 7
            elif length_data > 500:
                folds = 5
            else:
                folds = 3

            print(avg(cross_val_score(clf, data, target, cv=folds)), avg(cross_val_score(clf1, data, target, cv=folds)),
                  avg(cross_val_score(clf2, data, target, cv=folds)))
