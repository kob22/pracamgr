from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from simplefunctions import avarage_score

def runbagging(data,target):

    folds= [4,7,10]
    estimators = [10, 20, 50, 100, 200]
    for fold in folds:
        print('fold = %d ' % fold)
        for estimator in estimators:
            bagging = BaggingClassifier(tree.DecisionTreeClassifier(max_depth=10), max_samples=0.5, max_features=0.5, n_estimators=estimator)
            kf = KFold(len(target), n_folds=fold)
            print(avarage_score(cross_validation.cross_val_score(bagging, data, target, cv=kf, n_jobs=-1)))
