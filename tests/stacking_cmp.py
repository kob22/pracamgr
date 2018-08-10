from data import importdata
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from classifiers.stackingcv import StackingCVClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document, Section
from pylatex.utils import bold
from classifiers.stacking import StackingClassifier
from classifiers.stackingcv import StackingCVClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import os

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['seeds', 'new_thyroid', 'vehicle', 'ionosphere', 'vertebal', 'yeastME3', 'ecoli', 'bupa',
           'horse_colic',
           'german', 'breast_cancer', 'cmc', 'hepatitis', 'haberman', 'transfusion',
           'car', 'glass', 'abalone16_29', 'solar_flare', 'heart_cleveland', 'balance_scale', 'postoperative']

sections = ["Accuracy", "Sensitivity", "Specificity", "F-1 klasa mniejszosciowa", 'G-mean']

random_state = 5
tables = []
for tab in range(5):
    table = Tabular('c|cccccc')
    table.add_row(('', "KNN", "TREE", "NB", "STK", "STK PROBA", "VOTING"))
    table.add_hline()
    tables.append(table)

# liczba powtorzen klasyfikacji
iterations = 10

# liczba fold w sprawdzianie krzyzowym
folds = 10

# klasyfikatory
clf1 = KNeighborsClassifier()
clf2 = tree.DecisionTreeClassifier()
clf3 = GaussianNB()
meta = MLPClassifier(solver='lbfgs', random_state=1)
# glosowanie wiekszosciowe
voting = VotingClassifier(
    estimators=[('KNN', KNeighborsClassifier()), ('TREE', tree.DecisionTreeClassifier()), ('NB', GaussianNB())])

# stacking
sclf = StackingClassifier(
    classifiers=[KNeighborsClassifier(), tree.DecisionTreeClassifier(), GaussianNB()],
    meta_classifier=meta)

sclfproba = StackingClassifier(
    classifiers=[KNeighborsClassifier(), tree.DecisionTreeClassifier(), GaussianNB()],
    meta_classifier=meta, use_probas=True)

clfs = [clf1, clf2, clf3, sclf, sclfproba, voting]
for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)
    importdata.print_info(db.target)
    rows = []
    for i in range(5):
        rows.append([data])

    # obliczenia dla kazdego klasyfikatora
    for clf in clfs:
        scores = []
        # powtarzanie klasyfikacji
        for iteration in range(iterations):
            clf_ = clone(clf)
            # sprawdzian krzyzowy
            testpredict, testtarget = cross_val_pred2ict(clf_, db.data, db.target, cv=folds, n_jobs=-1)
            scores.append(accsespf1g(testpredict, testtarget))
            print(str(clf))
            print_scores(testpredict, testtarget)
        # usrednanie wynikow
        avgscores = avgaccsespf1g(scores)
        to_decimal = print_to_latex_two_decimal(avgscores)
        for i, score in enumerate(to_decimal):
            rows[i].append(score)
    # dodanie do tabeli
    for table, row in zip(tables, rows):
        max_v = max(row[1:])
        new_row = []
        for item in row:
            if item == max_v:
                new_row.append(bold(max_v))
            else:
                new_row.append(item)
        table.add_row(new_row)


# zapis do pliku
doc = Document("Stacking")
for i, tab, in enumerate(tables):
    section = Section(sections[i])
    section.append(tab)
    doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
