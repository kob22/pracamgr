from imblearn.combine import SMOTEENN
from sklearn.metrics import roc_auc_score
from sklearn import tree
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from data import importdata
from sklearn.preprocessing import StandardScaler
from pylatex import Tabular, Document, Section, MultiColumn, Subsection
import os

path = os.path.dirname(os.path.abspath(__file__))

# klasyfkatory
clf1 = tree.DecisionTreeClassifier()
clf2 = GaussianNB()
clf3 = KNeighborsClassifier()
clf4 = svm.SVC(kernel='rbf', probability=True)

clfs_normal = [('Decision Tree', clf1), ('NB', clf2)]
clfs_stand = [('kNN', clf3), ('SVM', clf4)]


# bez oversamplingu
def cross_val_oversampling_before(data, target):
    # skalowanie dla SVM i kNN
    stdsc = StandardScaler()
    datastdsc = stdsc.fit_transform(data)

    # podzial CV w zaleznosci od ilosci probek
    length_data = len(data)
    if length_data > 1000:
        folds = 10
    elif length_data > 700:
        folds = 7
    elif length_data > 500:
        folds = 5
    else:
        folds = 3

    rows_normal = []
    rows_stand = []

    for clf in clfs_normal:
        print(clf[0])
        clf_ = clone(clf[1])
        testpredict, testtarget = cross_val_pred2ict(clf_, data, target, cv=folds,
                                                     n_jobs=-1)
        print_scores(testpredict, testtarget)
        row = []
        row.extend(print_to_latex_sespf1g(testpredict, testtarget))

        # roc
        testroc = cross_val_predict(clf_, data, target, cv=folds,
                                    n_jobs=-1, method='predict_proba')
        row.append(float("{0:.2f}".format(roc_auc_score(y_true=target, y_score=testroc[:, 1]))))

        rows_normal.extend(row)

    for clf in clfs_stand:
        print(clf[0])
        clf_ = clone(clf[1])
        testpredict, testtarget = cross_val_pred2ict(clf_, datastdsc, target, cv=folds,
                                                     n_jobs=-1)
        print_scores(testpredict, testtarget)
        row = []
        row.extend(print_to_latex_sespf1g(testpredict, testtarget))

        # roc
        testroc = cross_val_predict(clf_, datastdsc, target, cv=folds,
                                    n_jobs=-1, method='predict_proba')
        row.append(float("{0:.2f}".format(roc_auc_score(y_true=target, y_score=testroc[:, 1]))))
        rows_stand.extend(row)

    return rows_normal, rows_stand


def cross_val_oversampling_wrong(data, target):
    # algorytm smoteenn
    sm = SMOTEENN()

    # podzial danych na uczace i testowe, 10% to dane testowe
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=5, stratify=target)

    X_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    stdsc = StandardScaler()
    datastdsc = stdsc.fit_transform(X_resampled)
    datastdsc_test = stdsc.transform(X_test)

    length_data = len(data)
    if length_data > 1000:
        folds = 10
    elif length_data > 700:
        folds = 7
    elif length_data > 500:
        folds = 5
    else:
        folds = 3
    rows_normal = [[], []]
    rows_stand = [[], []]
    for clf, row in zip(clfs_normal, rows_normal):
        print(clf[0])
        clf_ = clone(clf[1])
        testpredict, testtarget = cross_val_pred2ict(clf_, X_resampled, y_resampled, cv=folds,
                                                     n_jobs=-1)
        print_scores(testpredict, testtarget)
        row_temp = []
        row_temp.extend(print_to_latex_sespf1g(testpredict, testtarget))

        # roc
        testroc = cross_val_predict(clf_, X_resampled, y_resampled, cv=folds,
                                    n_jobs=-1, method='predict_proba')
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=y_resampled, y_score=testroc[:, 1]))))

        print("Testowanie oversamplingu na prawdziwych danych")
        clf_.fit(X_resampled, y_resampled)
        predict_clf_ = clf_.predict(X_test)

        print_scores([predict_clf_], [y_test])
        row_temp.extend(print_to_latex_sespf1g([predict_clf_], [y_test]))

        # roc
        test_roc = clf_.predict_proba(X_test)
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=y_test, y_score=test_roc[:, 1]))))
        row.extend(row_temp)

    for clf, row in zip(clfs_stand, rows_stand):
        print(clf[0])
        clf_ = clone(clf[1])
        testpredict, testtarget = cross_val_pred2ict(clf_, datastdsc, y_resampled, cv=folds,
                                                     n_jobs=-1)
        print_scores(testpredict, testtarget)
        row_temp = []
        row_temp.extend(print_to_latex_sespf1g(testpredict, testtarget))

        # roc
        testroc = cross_val_predict(clf_, datastdsc, y_resampled, cv=folds,
                                    n_jobs=-1, method='predict_proba')
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=y_resampled, y_score=testroc[:, 1]))))

        print("Testowanie oversamplingu na prawdziwych danych")
        clf_.fit(datastdsc, y_resampled)
        predict_clf_ = clf_.predict(datastdsc_test)

        print_scores([predict_clf_], [y_test])
        row_temp.extend(print_to_latex_sespf1g([predict_clf_], [y_test]))

        # roc
        test_roc = clf_.predict_proba(datastdsc_test)
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=y_test, y_score=test_roc[:, 1]))))
        row.extend(row_temp)

    return rows_normal, rows_stand


# dobrze zrobiony oversampling z cross validation
def cross_val_oversampling_correct(data, target):
    # algorytm smoteenn
    sm = SMOTEENN()

    # skalowanie dla SVM i kNN
    stdsc = StandardScaler()
    datastdsc = stdsc.fit_transform(data)

    # podzial CV w zaleznosci od ilosci probek
    length_data = len(data)
    if length_data > 1000:
        folds = 10
    elif length_data > 700:
        folds = 7
    elif length_data > 500:
        folds = 5
    else:
        folds = 3

    skf = StratifiedKFold(n_splits=folds, random_state=5)
    rows_normal = []
    rows_stand = []

    for clf in clfs_normal:
        print(clf[0])
        predict_re = []
        targets_re = []
        proba_re = []
        target_proba_re = []
        for train_index, test_index in skf.split(data, target):
            clf_train_ = clone(clf[1])
            data_re, tar_re = sm.fit_sample(data[train_index], target[train_index])
            clf_train_.fit(data_re, tar_re)
            predict_re.append(clf_train_.predict(data[test_index]))
            targets_re.append(target[test_index])
            proba_re.extend(clf_train_.predict_proba(data[test_index])[:, 1])
            target_proba_re.extend(target[test_index])

        print_scores(predict_re, targets_re)
        row_temp = []
        row_temp.extend(print_to_latex_sespf1g(predict_re, targets_re))
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=target_proba_re, y_score=proba_re))))
        rows_normal.extend(row_temp)

    for clf in clfs_stand:
        print(clf[0])
        predict_re = []
        targets_re = []
        proba_re = []
        target_proba_re = []
        for train_index, test_index in skf.split(data, target):
            clf_train_ = clone(clf[1])
            data_re, tar_re = sm.fit_sample(data[train_index], target[train_index])
            stdsc = StandardScaler()
            datastdsc = stdsc.fit_transform(data_re)
            datastdsc_test = stdsc.transform(data[test_index])
            clf_train_.fit(datastdsc, tar_re)
            predict_re.append(clf_train_.predict(datastdsc_test))
            targets_re.append(target[test_index])
            proba_re.extend(clf_train_.predict_proba(datastdsc_test)[:, 1])
            target_proba_re.extend(target[test_index])

        print_scores(predict_re, targets_re)
        row_temp = []
        row_temp.extend(print_to_latex_sespf1g(predict_re, targets_re))
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=target_proba_re, y_score=proba_re))))
        rows_stand.extend(row_temp)

    return rows_normal, rows_stand


doc = Document("test_cv_oversampling")
dataset = ['abalone0_4', 'abalone041629']

print("Dane bez oversamplingu")
names = ['Decision Tree', 'Naive Bayes', 'kNN', 'SVM']
tables = []
for tab in (0, 2):
    table = Tabular('|c|c|c|c|c|c|c|c|c|c|c|')
    table.add_hline()
    table.add_row('', MultiColumn(5, align='|c|', data=names[tab]), MultiColumn(5, align='|c|', data=names[tab + 1]))
    table.add_hline(start=2)
    table.add_row(('', "Se", "Sp", "F1", "G", "AUC", "Se", "Sp", "F1", "G", "AUC"))
    table.add_hline()
    tables.append(table)

section = Section("Dane bez oversamplingu")
subsections = [Subsection('Decision tree i Naive bayes'), Subsection('kNN i SVM')]
for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("DANE %s" % data)
    rows = []
    for i in range(2):
        rows.append([data])

    rets = cross_val_oversampling_before(db.data, db.target)

    for ret, row in zip(rets, rows):
        row.extend(ret)

    for table, row in zip(tables, rows):
        table.add_row(row)

for table, subsection in zip(tables, subsections):
    table.add_hline()
    subsection.append(table)
    section.append(subsection)

doc.append(section)

print("Dane z oversampling, z blednie zrobiona CV")
section2 = Section("Dane z oversampling, z blednie zrobiona CV")
subsections2 = [Subsection('Decision tree'), Subsection('Naive bayes'), Subsection('kNN'), Subsection('SVM')]

names2 = ['Decision Tree', 'Decision Tree TEST', 'Naive Bayes', 'Naive Bayes TEST', 'kNN', 'kNN TEST', 'SVM',
          'SVM TEST']
tables2 = []
for tab in (0, 2, 4, 6):
    table = Tabular('|c|c|c|c|c|c|c|c|c|c|c|')
    table.add_hline()
    table.add_row('', MultiColumn(5, align='|c|', data=names2[tab]), MultiColumn(5, align='|c|', data=names2[tab + 1]))
    table.add_hline(start=2)
    table.add_row(('', "Se", "Sp", "F1", "G", "AUC", "Se", "Sp", "F1", "G", "AUC"))
    table.add_hline()
    tables2.append(table)

for data in dataset:
    print("DANE %s" % data)
    db = getattr(importdata, 'load_' + data)()
    rows = []
    for i in range(4):
        rows.append([data])

    rets = cross_val_oversampling_wrong(db.data, db.target)
    i = 0
    for ret in rets:
        rows[i].extend(ret[0])
        rows[i + 1].extend(ret[1])
        i += 2

    for table, row in zip(tables2, rows):
        table.add_row(row)

for table, subsection in zip(tables2, subsections2):
    table.add_hline()
    subsection.append(table)
    section2.append(subsection)

doc.append(section2)

print("Cross validation z poprawnie zrobionym oversamplingiem")
names3 = ['Decision Tree', 'Naive Bayes', 'kNN', 'SVM']
tables3 = []
for tab in (0, 2):
    table = Tabular('|c|c|c|c|c|c|c|c|c|c|c|')
    table.add_hline()
    table.add_row('', MultiColumn(5, align='|c|', data=names3[tab]), MultiColumn(5, align='|c|', data=names3[tab + 1]))
    table.add_hline(start=2)
    table.add_row(('', "Se", "Sp", "F1", "G", "AUC", "Se", "Sp", "F1", "G", "AUC"))
    table.add_hline()
    tables3.append(table)

section3 = Section("Cross validation z poprawnie zrobionym oversamplingiem")
subsections3 = [Subsection('Decision tree i Naive bayes'), Subsection('kNN i SVM')]
for data in dataset:
    print("DANE %s" % data)
    db = getattr(importdata, 'load_' + data)()
    rows = []
    for i in range(2):
        rows.append([data])

    rets = cross_val_oversampling_correct(db.data, db.target)

    for ret, row in zip(rets, rows):
        row.extend(ret)

    for table, row in zip(tables3, rows):
        table.add_row(row)

for table, subsection in zip(tables3, subsections3):
    table.add_hline()
    subsection.append(table)
    section3.append(subsection)

doc.append(section3)

doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
