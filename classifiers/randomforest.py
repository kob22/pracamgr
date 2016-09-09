from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from simplefunctions import avarage_score

def runforest(data,target):

    folds= [4,7,10]
    estimators = [10,20,50,100,200]
    print("------------ RANDOM FOREST  ------------")

    for fold in folds:
        kf = KFold(len(target), n_folds=fold)
        print('fold = %d ' % fold)
        for estimator in estimators:
            clf = RandomForestClassifier(n_estimators=estimator)

            print(avarage_score([clf.fit(data[train], target[train]).score(data[test], target[test]) for train, test in kf]))
            print(avarage_score(cross_validation.cross_val_score(clf, data, target, cv=kf, n_jobs=-1)))