from data import importdata
import numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity

dataset = ['abalone16_29', 'balance_scale', 'breast_cancer', 'car', 'cmc',
           'ecoli', 'glass', 'haberman', 'heart_cleveland', 'hepatitis',
           'new_thyroid', 'postoperative', 'solar_flare', 'transfusion', 'vehicle',
           'yeastME3', 'bupa', 'german', 'horse_colic', 'ionosphere', 'seeds', 'vertebal']

for data in dataset:
    db = getattr(importdata, 'load_' + data)()
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Zbior danych: %s' % data)

    metrics = ['minkowski']
    for metric in metrics:
        nearestN = KernelDensity(kernel='epanechnikov')

        nearestN.fit(db.data, db.target)

        miniority_ind = np.where(db.target == 1)
        miniority_data = db.data[miniority_ind]
        miniority_target = db.target[miniority_ind]
        for d in miniority_data:
            print(nearestN.score(d))
