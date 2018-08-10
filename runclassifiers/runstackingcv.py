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
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from classifiers.stackingcv import StackingCVClassifier


def runstacking(data, target):
    folds = [10]

    for fold in folds:
        print('fold = %d ' % fold)

        clf1 = KNeighborsClassifier(n_neighbors=5)
        clf2 = tree.DecisionTreeClassifier(random_state=1)
        clf3 = GaussianNB()
        lr = LogisticRegression(C=10.0)

        # skf = StratifiedKFold(n_splits=fold, random_state=2)
        eclf1 = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
        eclf1.fit(data, target)

        print_scores([eclf1.predict(data)], [target])
        a = [data[0], data[1]]
        print(eclf1.predict(a))
