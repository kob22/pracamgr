from sklearn import tree
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from simplefunctions import *
from cross_val.cross_val import cross_val_predi2ct


def runtree(data,target):
    # kf = KFold(len(target), n_folds=4)
    # clf = tree.DecisionTreeClassifier(max_depth=5)
    #
    # print([clf.fit(data[train], target[train]).score(data[test], target[test]) for train, test in kf])
    # print(cross_validation.cross_val_score(clf, data, target, cv=kf, n_jobs=-1))
    #
    # kf = KFold(len(target), n_folds=7)
    # clf = tree.DecisionTreeClassifier(max_depth=5)
    #
    # print([clf.fit(data[train], target[train]).score(data[test], target[test]) for train, test in kf])
    # print(cross_validation.cross_val_score(clf, data, target, cv=kf, n_jobs=-1))

    folds= [4]
    depths = [5]
    print("------------ TREE ------------")

    for fold in folds:
        matrix = []
        kf = KFold(len(target), n_folds=fold)
        print('fold = %d ' % fold)
        for depth in depths:
            clf = tree.DecisionTreeClassifier()


            #print(avarage_score([clf.fit(data[train], target[train]).score(data[test], target[test]) for train, test in kf]))
            print(avarage_score(cross_validation.cross_val_score(clf, data, target, cv=kf, n_jobs=-1, scoring='f1')))
            testpredict,testtarget = cross_val_predi2ct(clf, data, target, cv=kf, n_jobs=-1)
            if len(testpredict) != len(testtarget):
                raise ValueError('length score and target are different!')
            for pr, tar in zip(testpredict, testtarget):
                matrix.append(confusion_matrix(tar, pr))
        print(matrix)
        prec = precision(matrix)
        recall = sensitivity(matrix)
        print(accuracy(matrix))
        print(prec)
        print(recall)
        print(f1tpfp(matrix))
        print(f1prre(prec[0], recall[0]))
        print(f1avg(matrix))

