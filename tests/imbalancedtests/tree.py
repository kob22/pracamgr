from data import importdata
from imbalanced import smoteentree_overfitting, smotentree

# dataset = ['load_breast_cancer','load_german', 'load_car', 'load_cmc']
dataset = ['abalone0_4', 'abalone041629', 'abalone16_29', 'balance_scale', 'breast_cancer', 'bupa', 'car', 'cmc',
           'ecoli', 'german', 'glass', 'haberman', 'heart_cleveland', 'hepatitis', 'horse_colic', 'ionosphere',
           'new_thyroid', 'postoperative', 'seeds', 'solar_flare', 'transfusion', 'vehicle', 'vertebal', 'yeastME1',
           'yeastME2', 'yeastME3']
for data in dataset:
    print("-------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------")
    print(data)
    db = getattr(importdata, 'load_' + data)()
    importdata.print_info(db.target)
    smoteentree_overfitting.runtree(db.data, db.target)
    # smotentree.runtree(db.data, db.target)
