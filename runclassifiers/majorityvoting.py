from sklearn.ensemble import VotingClassifier
from simplefunctions import *
from cross_val.cross_val import cross_val_pred2ict
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def runvoting(data, target):
    folds = [10]

    for fold in folds:
        print('fold = %d ' % fold)

        clf1 = KNeighborsClassifier(n_neighbors=5)
        clf2 = tree.DecisionTreeClassifier(random_state=1)
        clf3 = GaussianNB()
        skf = StratifiedKFold(n_splits=fold, random_state=2)
        eclf1 = VotingClassifier(estimators=[('Ada', clf1), ('RandomForest', clf2), ('SVM', clf3)], voting='hard')
        eclf2 = VotingClassifier(estimators=[('Ada', clf1), ('RandomForest', clf2), ('SVM', clf3)], voting='soft')
        for clf, label in zip([clf1, clf2, clf3, eclf1],
                              ['Ada', 'RandomForest', 'SVM RBF', 'ESEMBLE HARD', 'ESEMBLE SOFT']):
            testpredict, testtarget = cross_val_pred2ict(clf, data, target, cv=skf.get_n_splits(data, target),
                                                         n_jobs=-1)
            print("--------------------------")
            print(label)
            print_scores(testpredict, testtarget)
        eclf1.fit(data, target)