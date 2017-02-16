from data import importdata
from tests.gridsearch.tree import runtreegrid
import numpy as np
#from sklearn import tree
from runclassifiers import runmy, runmycv, tree, majorityvoting, bagging, svm, runnaivebayes, randomforest, runknn

#german = importdata.importfile("files/german.data-numeric")
# german = importdata.importfile("files/bupa.data")
# german = importdata.importfile("files/haberman.data")
# german = importdata.importfile("files/ionosphere.data")
# german = importdata.importfile("files/transfusion.data")

dataset = ['abalone16_29', 'balance_scale', 'breast_cancer', 'car', 'cmc',
           'ecoli', 'glass', 'haberman', 'heart_cleveland', 'hepatitis',
           'new_thyroid', 'postoperative', 'solar_flare', 'transfusion', 'vehicle',
           'yeastME3', 'bupa', 'german', 'horse_colic', 'ionosphere', 'seeds', 'vertebal']

for data in dataset:
    print("-------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------")

    db = getattr(importdata, 'load_' + data)()
    importdata.print_info(db.target)

    # runknn.runKNN(db.data,db.target)
    # importdata.print_info(db.target)
    # majorityvoting.runvoting(db.data, db.target)
    #runmy.runsmy(db.data, db.target)
    runnaivebayes.runNB(db.data, db.target)

    # rungrid(db.data, db.target)
    #tree.runtree(db.data, db.target)
    # bagging.runbaggingtree(db.data, db.target)
    #svm.runsvcn(db.data, db.target)
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
