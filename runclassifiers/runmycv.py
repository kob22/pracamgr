from classifiers.clf_expert import ensembel_rating
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


def runsmycv(data, target):
    folds = [10]

    for fold in folds:
        print('fold = %d ' % fold)

        clf1 = KNeighborsClassifier(n_neighbors=2)
        clf2 = tree.DecisionTreeClassifier(max_depth=6, random_state=1)
        clf3 = GaussianNB()
        clf2.fit(data, target)
        myclf = F1ClassifierCV(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)])

        myclf.fit(data, target)
        predict = myclf.predict(data)

        print_scores([np.asarray(predict)], [target])

        voting = VotingClassifier(estimators=[('KNN', clf1), ('TREE', clf2), ('NB', clf3)], voting='hard')
        voting.fit(data, target)
        predictv = voting.predict(data)
        print_scores([np.asarray(predictv)], [target])

        for clf, label in zip([clf1, clf2, clf3], ['KNN', 'TREE', 'NB']):
            print(label)
            clf.fit(data, target)
            predictions = clf.predict(data)
            print_scores([np.asarray(predictions)], [target])
