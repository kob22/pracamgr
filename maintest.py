from data import importdata
#from sklearn import tree
from runclassifiers import runmy, runmycv, tree

#german = importdata.importfile("files/german.data-numeric")
# german = importdata.importfile("files/bupa.data")
# german = importdata.importfile("files/haberman.data")
# german = importdata.importfile("files/ionosphere.data")
# german = importdata.importfile("files/transfusion.data")
dataset = ['load_german', 'load_haberman', 'load_transfusion', 'load_ionosphere']
for data in dataset:
    print("-------------------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------")
    db = getattr(importdata, data)()
    importdata.print_info(db.target)
    tree.runtree(db.data, db.target)

db = importdata.load_transfusion()
importdata.print_info(db.target)
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
