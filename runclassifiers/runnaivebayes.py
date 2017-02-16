from sklearn.naive_bayes import GaussianNB
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def runNB(data, target):
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
            nvb = GaussianNB()
            testpredict, testtarget = cross_val_pred2ict(nvb, data, target, cv=10,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)
            nvb = GaussianNB()
            testpredict, testtarget = cross_val_pred2ict(nvb, datamms, target, cv=10,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)
            nvb = GaussianNB()
            testpredict, testtarget = cross_val_pred2ict(nvb, datastdsc, target, cv=10,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)