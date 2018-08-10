from data import importdata
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.base import clone
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from pylatex import Tabular, Document, Section
from pylatex.utils import bold
from pylatex.basic import TextColor

import os

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['seeds', 'new_thyroid', 'vehicle', 'ionosphere', 'vertebal', 'yeastME3', 'ecoli', 'bupa',
           'horse_colic',
           'german', 'breast_cancer', 'cmc', 'hepatitis', 'haberman', 'transfusion',
           'car', 'glass', 'abalone16_29', 'solar_flare', 'heart_cleveland', 'balance_scale', 'postoperative']

sections = ["Accuracy", "Sensitivity", "Specificity", "F-1 klasa mniejszosciowa", 'G-mean']
random_state = 5

# ustawienia baggingu
samples = [0.68]
features = [0.72]

# liczba powtorzen klasyfikacji
iterations = 10

# liczba fold w sprawdzianie krzyzowym
folds = 10
for feat in features:
    for samp in samples:
        tables = []
        for tab in range(5):
            table = Tabular('c|cccccccc')
            table.add_hline()
            table.add_row(('', "NB", "5", "10", "15", "30", "50", "100", "200"))
            table.add_hline()
            tables.append(table)

        # klasyfikatory
        clf1 = GaussianNB()
        clfs = [clf1,
                BaggingClassifier(GaussianNB(), n_estimators=5, max_samples=samp, max_features=feat),
                BaggingClassifier(GaussianNB(), n_estimators=10, max_samples=samp, max_features=feat),
                BaggingClassifier(GaussianNB(), n_estimators=15, max_samples=samp, max_features=feat),
                BaggingClassifier(GaussianNB(), n_estimators=30, max_samples=samp, max_features=feat),
                BaggingClassifier(GaussianNB(), n_estimators=50, max_samples=samp, max_features=feat),
                BaggingClassifier(GaussianNB(), n_estimators=100, max_samples=samp, max_features=feat),
                BaggingClassifier(GaussianNB(), n_estimators=200, max_samples=samp, max_features=feat)]

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
                #powtarzanie klasyfikacji
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

            for table, row in zip(tables, rows):
                print(row)
                max_v = max(row[1:])
                new_row = []

                for item in row:
                    if item == max_v:
                        new_row.append(bold(max_v))
                    else:
                        new_row.append(item)
                table.add_row(new_row)
        # zapis do pliku
        doc = Document("bagging_NB%s%s" % (feat, samp))
        for i, tab, in enumerate(tables):
            section = Section(sections[i])
            section.append(tab)
            doc.append(section)
        doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
        doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
