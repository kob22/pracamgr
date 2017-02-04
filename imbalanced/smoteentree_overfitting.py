from imblearn.combine import SMOTEENN
from sklearn import tree
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
## zostawic to, pokaze sie jako blad takiego postepowania
################################### PAMIETAC O SKALOWANIU $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
################################### PAMIETAC O SKALOWANIU $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
################################### PAMIETAC O SKALOWANIU $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
################################### PAMIETAC O SKALOWANIU $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
################################### PAMIETAC O SKALOWANIU $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def runtree(data, target):

    sm = SMOTEENN()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=5, stratify=target)
    print(y_test.size)
    print(np.bincount(y_test))
    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    folds = [10]
    depths = [10]
    print("------------ TREE ------------")

    for fold in folds:
        print('fold = %d ' % fold)
        for depth in depths:
            print('depth = %d ' % depth)
            clf = GaussianNB()
            skf = StratifiedKFold(n_splits=fold, random_state=5)

            testpredict, testtarget = cross_val_pred2ict(clf, data, target, cv=fold,
                                                         n_jobs=-1)

            print_scores(testpredict, testtarget)
            print('smotens - przecenione')
            testpredict, testtarget = cross_val_pred2ict(clf, X_resampled, y_resampled,
                                                         cv=fold,
                                                         n_jobs=-1)
            print_scores(testpredict, testtarget)

            testpredict = cross_val_predict(clf, data, target, cv=fold,
                                            n_jobs=-1, method='predict_proba')
            print(roc_auc_score(y_true=target, y_score=testpredict[:, 1]))

            testpredict = cross_val_predict(clf, X_resampled, y_resampled,
                                            cv=fold,
                                            n_jobs=-1, method='predict_proba')
            print(roc_auc_score(y_true=y_resampled, y_score=testpredict[:, 1]))
            print('smotens - na czesci')
            clf.fit(X_resampled, y_resampled)
            print_scores([clf.predict(X_test)], [y_test])
            # print(roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1]))

            print('smotens - wlasciwe')
            clf_train = KNeighborsClassifier()
            predict_re = []
            targets_re = []
            proba_re = []
            target_proba_re = []

            for train_index, test_index in skf.split(data, target):
                clf_train_ = clone(clf_train)
                data_re, tar_re = sm.fit_sample(data[train_index], target[train_index])
                clf_train_.fit(data_re, tar_re)
                predict_re.append(clf_train_.predict(data[test_index]))
                targets_re.append(target[test_index])
                proba_re.extend(clf_train_.predict_proba(data[test_index])[:, 1])
                target_proba_re.extend(target[test_index])

            print_scores(predict_re, targets_re)
            # print(test_re)
            # print(proba_re)
            print(roc_auc_score(y_true=target_proba_re, y_score=proba_re))
