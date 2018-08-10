from sklearn.neighbors import KNeighborsClassifier
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def runKNN(data, target):
    folds = [10]
    depths = [10]
    print("------------ NB ------------")
    mms = MinMaxScaler()
    stdsc = StandardScaler()
    datamms = mms.fit_transform(data)
    datastdsc = stdsc.fit_transform(data)
    for fold in folds:
        print('fold = %d ' % fold)
        for depth in depths:
            knn = KNeighborsClassifier()
            testpredict, testtarget = cross_val_pred2ict(knn, data, target, cv=10,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)

            testpredict, testtarget = cross_val_pred2ict(knn, datamms, target, cv=10,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)
            testpredict, testtarget = cross_val_pred2ict(knn, datastdsc, target, cv=10,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)
