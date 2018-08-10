from classifiers.stacking import StackingClassifier
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


def runstacking(data, target):
    folds = [10]

    for fold in folds:
        print('fold = %d ' % fold)

        clf1 = KNeighborsClassifier(n_neighbors=5)
        clf2 = tree.DecisionTreeClassifier(random_state=1)
        clf3 = GaussianNB()
        lr = LogisticRegression(C=10.0)
        sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                                  meta_classifier=lr)

        params = {'kneighborsclassifier__n_neighbors': [1, 5],
                  'decisiontreeclassifier__max_depth': [1, 10, 50],
                  'meta-logisticregression__C': [0.1, 10.0]}

        grid = GridSearchCV(estimator=sclf,
                            param_grid=params,
                            cv=10,
                            refit=True)
        grid.fit(data, target)

        cv_keys = ('mean_test_score', 'std_test_score', 'params')

        for r, _ in enumerate(grid.cv_results_['mean_test_score']):
            print("%0.3f +/- %0.2f %r"
                  % (grid.cv_results_[cv_keys[0]][r],
                     grid.cv_results_[cv_keys[1]][r] / 2.0,
                     grid.cv_results_[cv_keys[2]][r]))

        print('Best parameters: %s' % grid.best_params_)
        print('Accuracy: %.2f' % grid.best_score_)

        skf = StratifiedKFold(n_splits=fold, random_state=2)
        eclf1 = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
        for clf, label in zip([clf1, clf2, clf3, eclf1],
                              ['Ada', 'RandomForest', 'SVM RBF', 'ESEMBLE HARD', 'ESEMBLE SOFT']):
            testpredict, testtarget = cross_val_pred2ict(clf, data, target, cv=skf.get_n_splits(data, target),
                                                         n_jobs=-1)
            print("--------------------------")
            print(label)
            print_scores(testpredict, testtarget)
