from data import importdata
import numpy as np
#from sklearn import tree
from runclassifiers import runmy, runmycv, tree, majorityvoting, bagging, svm, runnaivebayes, randomforest

#german = importdata.importfile("files/german.data-numeric")
# german = importdata.importfile("files/bupa.data")
# german = importdata.importfile("files/haberman.data")
# german = importdata.importfile("files/ionosphere.data")
# german = importdata.importfile("files/transfusion.data")

dataset = ['load_german', 'load_haberman', 'load_transfusion', 'load_ionosphere', 'load_balance_scale', 'load_bupa',
           'load_car', 'load_cmc', 'load_ecoli',
           'load_glass', 'load_new_thyroid', 'load_seeds', 'load_solar_flare', 'load_vehicle', 'load_vertebal',
           'load_yeastME1', 'load_yeastME2', 'load_yeastME3',
           'load_abalone0_4', 'load_abalone16_29', 'load_abalone0_4_16_29']
# datasetmissing = ['load_hear_cleveland', 'load_hepatitis', 'load_breast_cancer', 'load_postoperative']
# dataset = [['load_breast_cancer', 'median'], ['load_hepatitis', 'median'], ['load_hear_cleveland', 'median'],
#           ['load_postoperative', 'median'], ['load_horse_colic', 'median']]
# dataset = ['load_breast_cancer','load_german', 'load_car', 'load_cmc']
for data in dataset:
    print("-------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------")
    db = getattr(importdata, data)()
    importdata.print_info(db.target)


        # svm.runsvcn(db.data,db.target)
        # importdata.print_info(db.target)
        # majorityvoting.runvoting(db.data, db.target)
        # runmycv.runsmycv(db.data, db.target)
    # runnaivebayes.runNB(db.data,db.target)

    tree.runtree(db.data, db.target)
    # bagging.runbaggingtree(db.data,db.target)
    #randomforest.runforest(db.data,db.target)

#clf = tree.DecisionTreeClassifier()
#a = clf.fit(german.data,german.target).predict(german.data)
#print (confusion_matrix(german.target, a))

# german.shuffle2()
#tree.runtree(german.data,german.target)
#svm.runsvcn(german.data,german.target)
#randomforest.runforest(german.data,german.target)
#bagging.runbaggingtree(german.data,german.target)
#adaboost.runada(german.data,german.target)
# majorityvoting.runvoting(german.data, german.target)
# runmy.runsmy(db.data,db.target)
# runmycv.runsmycv(db.data, db.target)
# runstackingcv.runstacking(german.data, german.target)

#moze portal ?
