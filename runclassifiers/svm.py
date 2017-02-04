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
            testpredict, testtarget = cross_val_pred2ict(svc, data, target, cv=10,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)

            testpredict, testtarget = cross_val_pred2ict(svc, datamms, target, cv=10,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)
            testpredict, testtarget = cross_val_pred2ict(svc, datastdsc, target, cv=10,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)


def runsvc(kernel, data, target, nfolds):
    skf = StratifiedKFold(n_splits=nfolds, random_state=5)
    svc = svm.SVC(C=1, kernel=kernel)
    start = time.time()
    ret = (cross_val_score(svc, data, target, cv=skf, n_jobs=-1))
    end = time.time()

    return ret, (end - start)
