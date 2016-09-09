from sklearn import svm, tree
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import time
from simplefunctions import avarage_score

def runsvcn(data,target):
    folds= [4,7,10]
    kernels = ['linear', 'rbf', 'sigmoid']
    print("------------ SVM  ------------")

    for fold in folds:
        kf = KFold(len(target), n_folds=fold)
        print('fold = %d ' % fold)
        for kernel in kernels:
            kf = KFold(len(target), n_folds=fold)
            svc = svm.SVC(C=1, kernel=kernel)

            print(avarage_score([svc.fit(data[train], target[train]).score(data[test], target[test]) for train, test in kf]))
            print(avarage_score(cross_validation.cross_val_score(svc, data, target, cv=kf, n_jobs=-1)))


def runsvc(kernel, data,target,nfolds):
    kf = KFold(len(target), n_folds=nfolds)
    svc = svm.SVC(C=1, kernel=kernel)
    start = time.time()
    ret = (cross_validation.cross_val_score(svc, data, target, cv=kf, n_jobs=-1, ))
    end = time.time()

    return ret, (end - start)
