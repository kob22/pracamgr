from data import importdata
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pylatex import Tabular, Document, Section
import os

path = os.path.dirname(os.path.abspath(__file__))
dataset = ['seeds', 'new_thyroid', 'vehicle', 'ionosphere', 'vertebal', 'yeastME3', 'yeastME1', 'ecoli', 'bupa',
           'horse_colic',
           'abalone0_4', 'german', 'breast_cancer', 'cmc', 'hepatitis', 'haberman', 'yeastME2', 'abalone041629',
           'transfusion',
           'car', 'glass', 'abalone16_29', 'solar_flare', 'heart_cleveland', 'balance_scale', 'postoperative']

table = Tabular('c|cccc')
table.add_row(('', "Safe [%]", "Borderline [%]", "Rare [%]", "Outlier [%]"))
table.add_hline()

for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)

    nearestN = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='chebyshev')

    nearestN.fit(db.data, db.target)

    miniority_ind = np.where(db.target == 1)
    miniority_data = db.data[miniority_ind]
    miniority_target = db.target[miniority_ind]

    sixNearest = nearestN.kneighbors(miniority_data, return_distance=False)
    neighbors_list = db.target[sixNearest]
    safe = 0
    border = 0
    outlier = 0
    rare = 0
    for target, neighbors in zip(miniority_target, neighbors_list):

        same_neighbors = np.count_nonzero(neighbors == target) - 1
        if same_neighbors > 3:
            safe += 1
        elif same_neighbors > 1:
            border += 1
        elif same_neighbors == 1:
            outlier += 1
        else:
            rare += 1

    count_all = np.count_nonzero(miniority_target)

    safeT = float("{0:.2f}".format((float(safe) / count_all) * 100))
    borderT = float("{0:.2f}".format((float(border) / count_all) * 100))
    rareT = float("{0:.2f}".format((float(rare) / count_all) * 100))
    outlierT = float("{0:.2f}".format((float(outlier) / count_all) * 100))

    table.add_row([data, safeT, borderT, rareT, outlierT])

doc = Document("analiza_danych")

section = Section("Analiza danych klasy mniejszosciowej")
section.append(table)
doc.append(section)
doc.generate_tex(os.path.join(path, 'wyniki/pliki_tex/'))
doc.generate_pdf(os.path.join(path, 'wyniki/pliki_pdf/'))
