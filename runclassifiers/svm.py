from sklearn import svm
from sklearn.metrics import confusion_matrix
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.model_selection._validation import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def runsvcn(data, target):
    folds = [10]
    kernels = ['rbf']
    print("------------ SVM  ------------")
    mms = MinMaxScaler()
    stdsc = StandardScaler()
    datamms = mms.fit_transform(data)
    datastdsc = stdsc.fit_transform(data)
    for fold in folds:
        print('fold = %d ' % fold)
        for kernel in kernels:
            print('----- KERNEL = %s -----' % kernel)
            matrices1 = []
            matrices2 = []
            skf = StratifiedKFold(n_splits=fold, random_state=5)

            svc = svm.SVC(C=1, kernel=kernel)
            svc.set_params()
            testpredict, testtarget = cross_val_pred2ict(svc, datastdsc, target, cv=skf.get_n_splits(datastdsc, target),
                                                         n_jobs=-1)

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


def runsvc(kernel, data, target, nfolds):
    skf = StratifiedKFold(n_splits=nfolds, random_state=5)
    svc = svm.SVC(C=1, kernel=kernel)
    start = time.time()
    ret = (cross_val_score(svc, data, target, cv=skf, n_jobs=-1))
    end = time.time()

    return ret, (end - start)
