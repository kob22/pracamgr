from data import importdata
from imbalanced import smoteentree_overfitting, smotentree

# dataset = ['load_breast_cancer','load_german', 'load_car', 'load_cmc']
dataset = ['load_german', 'load_haberman', 'load_transfusion', 'load_ionosphere', 'load_balance_scale', 'load_bupa',
           'load_car', 'load_cmc', 'load_ecoli',
           'load_glass', 'load_new_thyroid', 'load_seeds', 'load_solar_flare', 'load_vehicle', 'load_vertebal',
           'load_yeastME1', 'load_yeastME2', 'load_yeastME3',
           'load_abalone0_4', 'load_abalone16_29', 'load_abalone0_4_16_29']
for data in dataset:
    print("-------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------")
    db = getattr(importdata, data)()
    importdata.print_info(db.target)
    smoteentree_overfitting.runtree(db.data, db.target)
    # smotentree.runtree(db.data, db.target)
