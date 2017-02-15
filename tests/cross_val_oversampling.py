from imblearn.over_sampling import SMOTE
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
    print("Testowanie CV bez oversamplingu")
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
    # klasyfikator NB i tree
    for clf in clfs_normal:
        print('Klasyfikator: %s' % clf[0])
        clf_ = clone(clf[1])

        # CV
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

    # klasyfikator SVM i kNN
    for clf in clfs_stand:
        print('Klasyfikator: %s' % clf[0])
        clf_ = clone(clf[1])

        # CV
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
    print("Testowanie CV z wykonanym oversampling przed CV")
    # algorytm SMOTE
    sm = SMOTE()

    # podzial danych na uczace i testowe, 20% to dane testowe
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=5, stratify=target)

    # generowanie nowych probek (oversmapling) oraz standaryzacja dla SVM i kNN
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

    skf = StratifiedKFold(n_splits=folds, random_state=5)
    rows_normal = [[], []]
    rows_stand = [[], []]

    # klasyfikator tree i NB
    for clf, row in zip(clfs_normal, rows_normal):
        print('Klasyfikator: %s' % clf[0])
        predict_re = []
        targets_re = []
        proba_re = []
        target_proba_re = []
        predict_clf_test = []
        target_clf_test = []
        predict_clf_test_proba = []
        target_clf_test_proba = []

        # CV
        for train_index, test_index in skf.split(X_resampled, y_resampled):
            clf_train_ = clone(clf[1])

            # trenowanie
            clf_train_.fit(X_resampled[train_index], y_resampled[train_index])

            # testowanie
            predict_re.append(clf_train_.predict(X_resampled[test_index]))
            targets_re.append(y_resampled[test_index])
            proba_re.extend(clf_train_.predict_proba(X_resampled[test_index])[:, 1])
            target_proba_re.extend(y_resampled[test_index])

            # walidacja
            predict_clf_test.append(clf_train_.predict(X_test))
            target_clf_test.append(y_test)

            predict_clf_test_proba.extend(clf_train_.predict_proba(X_test)[:, 1])
            target_clf_test_proba.extend(y_test)

        print("Wyniki CV")
        # dodawanie do tabeli
        print_scores(predict_re, targets_re)
        row_temp = []
        row_temp.extend(print_to_latex_sespf1g(predict_re, targets_re))
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=target_proba_re, y_score=proba_re))))

        print("Wyniki walidacji CV")
        print_scores(predict_clf_test, target_clf_test)
        row_temp.extend(print_to_latex_sespf1g(predict_clf_test, target_clf_test))
        row_temp.append(
            float("{0:.2f}".format(roc_auc_score(y_true=target_clf_test_proba, y_score=predict_clf_test_proba))))
        row_temp.append(row_temp[3] - row_temp[8])
        row.extend(row_temp)

    # SVM i kNN
    for clf, row in zip(clfs_stand, rows_stand):
        print('Klasyfikator: %s' % clf[0])
        predict_re = []
        targets_re = []
        proba_re = []
        target_proba_re = []
        predict_clf_test = []
        target_clf_test = []
        predict_clf_test_proba = []
        target_clf_test_proba = []

        # CV
        for train_index, test_index in skf.split(datastdsc, y_resampled):
            clf_train_ = clone(clf[1])

            # trenowanie
            clf_train_.fit(datastdsc[train_index], y_resampled[train_index])

            # testowanie
            predict_re.append(clf_train_.predict(datastdsc[test_index]))
            targets_re.append(y_resampled[test_index])
            proba_re.extend(clf_train_.predict_proba(datastdsc[test_index])[:, 1])
            target_proba_re.extend(y_resampled[test_index])

            # walidacja
            predict_clf_test.append(clf_train_.predict(datastdsc_test))
            target_clf_test.append(y_test)

            predict_clf_test_proba.extend(clf_train_.predict_proba(datastdsc_test)[:, 1])
            target_clf_test_proba.extend(y_test)

        print("Wyniki CV")
        # dodawanie do tabeli
        print_scores(predict_re, targets_re)
        row_temp = []
        row_temp.extend(print_to_latex_sespf1g(predict_re, targets_re))
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=target_proba_re, y_score=proba_re))))

        print("Wyniki walidacji CV")
        print_scores(predict_clf_test, target_clf_test)
        row_temp.extend(print_to_latex_sespf1g(predict_clf_test, target_clf_test))
        row_temp.append(
            float("{0:.2f}".format(roc_auc_score(y_true=target_clf_test_proba, y_score=predict_clf_test_proba))))
        row_temp.append(row_temp[3] - row_temp[8])
        row.extend(row_temp)

    return rows_normal, rows_stand


# dobrze zrobiony oversampling z cross validation
def cross_val_oversampling_correct(data, target):
    print("Testowanie CV z wykonanym oversampling w trakcie CV")
    # algorytm smoteenn
    sm = SMOTE()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=5, stratify=target)

    # skalowanie dla SVM i kNN
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
    rows_normal = [[], []]
    rows_stand = [[], []]

    for clf, row in zip(clfs_normal, rows_normal):
        print('Klasyfikator: %s' % clf[0])
        predict_re = []
        targets_re = []
        proba_re = []
        target_proba_re = []
        predict_clf_test = []
        target_clf_test = []
        predict_clf_test_proba = []
        target_clf_test_proba = []

        # CV
        for train_index, test_index in skf.split(X_train, y_train):
            clf_train_ = clone(clf[1])

            # SMOTE
            data_re, tar_re = sm.fit_sample(X_train[train_index], y_train[train_index])

            # trenowanie
            clf_train_.fit(data_re, tar_re)

            # testowanie
            predict_re.append(clf_train_.predict(X_train[test_index]))
            targets_re.append(y_train[test_index])
            proba_re.extend(clf_train_.predict_proba(X_train[test_index])[:, 1])
            target_proba_re.extend(y_train[test_index])

            # walidacja
            predict_clf_test.append(clf_train_.predict(X_test))
            target_clf_test.append(y_test)

            predict_clf_test_proba.extend(clf_train_.predict_proba(X_test)[:, 1])
            target_clf_test_proba.extend(y_test)

        print("Wyniki CV")
        # dodawanie do tabeli
        print_scores(predict_re, targets_re)
        row_temp = []
        row_temp.extend(print_to_latex_sespf1g(predict_re, targets_re))
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=target_proba_re, y_score=proba_re))))

        print("Wyniki walidacji CV")
        print_scores(predict_clf_test, target_clf_test)
        row_temp.extend(print_to_latex_sespf1g(predict_clf_test, target_clf_test))
        row_temp.append(
            float("{0:.2f}".format(roc_auc_score(y_true=target_clf_test_proba, y_score=predict_clf_test_proba))))
        row_temp.append(row_temp[3] - row_temp[8])
        row.extend(row_temp)

    for clf, row in zip(clfs_stand, rows_stand):
        print('Klasyfikator: %s' % clf[0])
        predict_re = []
        targets_re = []
        proba_re = []
        target_proba_re = []
        predict_clf_test = []
        target_clf_test = []
        predict_clf_test_proba = []
        target_clf_test_proba = []
        for train_index, test_index in skf.split(X_train, y_train):
            clf_train_ = clone(clf[1])

            # SMOTE
            data_re, tar_re = sm.fit_sample(X_train[train_index], y_train[train_index])

            # standaryzacja
            stdsc = StandardScaler()
            datastdsc = stdsc.fit_transform(data_re)
            datastdsc_test = stdsc.transform(X_train[test_index])
            datastdsc_vad = stdsc.transform(X_test)

            # trenowanie
            clf_train_.fit(datastdsc, tar_re)

            # testowanie
            predict_re.append(clf_train_.predict(datastdsc_test))
            targets_re.append(y_train[test_index])
            proba_re.extend(clf_train_.predict_proba(datastdsc_test)[:, 1])
            target_proba_re.extend(y_train[test_index])

            # walidacja
            predict_clf_test.append(clf_train_.predict(datastdsc_vad))
            target_clf_test.append(y_test)

            predict_clf_test_proba.extend(clf_train_.predict_proba(datastdsc_vad)[:, 1])
            target_clf_test_proba.extend(y_test)

        print("Wyniki CV")
        # dodawanie do tabeli
        print_scores(predict_re, targets_re)
        row_temp = []
        row_temp.extend(print_to_latex_sespf1g(predict_re, targets_re))
        row_temp.append(float("{0:.2f}".format(roc_auc_score(y_true=target_proba_re, y_score=proba_re))))

        print("Wyniki walidacji CV")
        print_scores(predict_clf_test, target_clf_test)
        row_temp.extend(print_to_latex_sespf1g(predict_clf_test, target_clf_test))
        row_temp.append(
            float("{0:.2f}".format(roc_auc_score(y_true=target_clf_test_proba, y_score=predict_clf_test_proba))))
        row_temp.append(row_temp[3] - row_temp[8])
        row.extend(row_temp)

    return rows_normal, rows_stand


# dane
dataset = ['abalone0_4', 'abalone041629', 'abalone16_29', 'balance_scale', 'breast_cancer', 'bupa', 'car', 'cmc',
           'ecoli', 'german', 'glass', 'haberman', 'heart_cleveland', 'hepatitis', 'horse_colic', 'ionosphere',
           'new_thyroid', 'postoperative', 'seeds', 'solar_flare', 'transfusion', 'vehicle', 'vertebal', 'yeastME1',
           'yeastME2', 'yeastME3']

doc = Document("test_CV_oversampling")


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
    table = Tabular('|c|c|c|c|c|c|c|c|c|c|c|c|')
    table.add_hline()
    table.add_row('', MultiColumn(5, align='|c|', data=names2[tab]), MultiColumn(5, align='|c|', data=names2[tab + 1]),
                  '')
    table.add_hline(start=2)
    table.add_row(('', "Se", "Sp", "F1", "G", "AUC", "Se", "Sp", "F1", "G_t", "AUC", 'G-G_t'))
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
section3 = Section("Cross validation z poprawnie zrobionym oversamplingiem")
subsections3 = [Subsection('Decision tree'), Subsection('Naive bayes'), Subsection('kNN'), Subsection('SVM')]

names3 = ['Decision Tree', 'Decision Tree TEST', 'Naive Bayes', 'Naive Bayes TEST', 'kNN', 'kNN TEST', 'SVM',
          'SVM TEST']
tables3 = []
for tab in (0, 2, 4, 6):
    table = Tabular('|c|c|c|c|c|c|c|c|c|c|c|c|')
    table.add_hline()
    table.add_row('', MultiColumn(5, align='|c|', data=names3[tab]), MultiColumn(5, align='|c|', data=names3[tab + 1]),
                  '')
    table.add_hline(start=2)
    table.add_row(('', "Se", "Sp", "F1", "G", "AUC", "Se", "Sp", "F1", "G_t", "AUC", 'G-G_t'))
    table.add_hline()
    tables3.append(table)

for data in dataset:
    print("DANE %s" % data)
    db = getattr(importdata, 'load_' + data)()
    rows = []
    for i in range(4):
        rows.append([data])

    rets = cross_val_oversampling_correct(db.data, db.target)
    i = 0
    for ret in rets:
        rows[i].extend(ret[0])
        rows[i + 1].extend(ret[1])
        i += 2

    for table, row in zip(tables3, rows):
        table.add_row(row)

for table, subsection in zip(tables3, subsections3):
    table.add_hline()
    subsection.append(table)
    section3.append(subsection)

doc.append(section3)

doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
