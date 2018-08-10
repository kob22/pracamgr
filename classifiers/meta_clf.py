import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
import simplefunctions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from classifiers.clf_expert import clf_expert
from cross_val.cross_val import cross_val_pred2ict
from classifiers.stacking import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import StratifiedKFold

def _parallel_fit_estimator(estimator, X, y):
    """Private function used to fit an estimator within a job."""
    estimator.fit(X, y)
    return estimator


# METAK LASYFIKATOR
# funkcja porownujaca klasyfikatory moze byc: precision_tp_fp, g_meantpfp, f1tpfp
class meta_classifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    # inicjalizacja

    # function_compare - funkcja porownujaca klasyfikatory
    def __init__(self, estimators, estimators_bag, estimators_ada, n_jobs=1, function_compare='precision_tp_fp',
                 n_folds=3, n_estimators=100):
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.n_jobs = n_jobs
        self.groups = []
        self.g_mean = [-1, -1]
        self.function_compare = function_compare
        self.clfs = []
        self.n_folds = n_folds
        self.max_g = [-1, -1,-1]
        self.clf_id = [-1, -1, -1]
        self.n_estimators = n_estimators
        self.meta_clf_ = MLPClassifier(solver='lbfgs', random_state=1)
        self.clfs_ensemble = []
        self.estimators_bag = estimators_bag
        self.estimators_ada = estimators_ada
        self.random_st = 5
        self.methods = [SMOTE(k_neighbors=3,random_state=self.random_st),NeighbourhoodCleaningRule(n_neighbors=3, random_state=self.random_st)]
        self.methoda = [0,1]
        self.name_met = ["ADASYN", "NCR"]
        self.ensemble_ = []

    def fit(self, X, y):
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')
        cv_predictions = []
        targets = []

        # klonowanie
        self.estimators_ = [clone(estimator) for _, estimator in self.estimators]

        # dodawanie klasyfikatorow AdaBoost
        for clf in self.estimators_ada:
            self.clfs.append(AdaBoostClassifier(clone(clf), n_estimators=self.n_estimators))

        # dodawanie klasyfikatorow Bagging
        for clf in self.estimators_bag:
            self.clfs.append(BaggingClassifier(clone(clf), n_estimators=100, max_samples=0.9))

        self.clfs.append(StackingClassifier(classifiers=self.estimators_,
                                            meta_classifier=LogisticRegression()))
        self.clfs.append(clf_expert(self.estimators))
        # ocena klasyfikatorow
        for clf in self.clfs:
            testpredict, testtarget = cross_val_pred2ict(clf, X, y, cv=self.n_folds,
                                                         n_jobs=1)
            cv_predictions.append((testpredict))
            targets.append(testtarget)


        skf = StratifiedKFold(n_splits=2, random_state=self.random_st)

        # trenowanie i ocenianie klasyfiktorow dla zbioru SMOTE i NCR
        for clf in self.clfs:
            for method, name in zip(self.methoda, self.name_met):
                metodaa = SMOTE(k_neighbors=3,random_state=self.random_st)
                metodaj = NeighbourhoodCleaningRule(n_neighbors=3,random_state=self.random_st)


                predict_re = []
                targets_re = []
                for train_index, test_index in skf.split(X, y):

                    if method == 0:
                        data_re, tar_re = metodaa.fit_sample(np.asarray(X[train_index]), np.asarray(y[train_index]))
                    else:
                        data_re, tar_re = metodaj.fit_sample(np.asarray(X[train_index]), np.asarray(y[train_index]))

                    clf_ = clone(clf)

                        # trenowanie
                    clf_.fit(data_re, tar_re)

                        # testowanie
                    predict_re.append(clf_.predict(X[test_index]))
                    targets_re.append(y[test_index])
                cv_predictions.append((predict_re))
                targets.append(targets_re)


        # wylanianie 2 najlepszych ekspertow
        for idx, (prediction, target) in enumerate(zip(cv_predictions, targets)):

            matrixes1 = []
            matrixes2 = []
            for pred, tar in zip(prediction, target):
                matrixes1.append(simplefunctions.confusion_matrix(tar, pred))
            for matrix in matrixes1:
                matrixes2.append(np.array([[matrix[1, 1], matrix[1, 0]], [matrix[0, 1], matrix[0, 0]]]))
            fun_cmp = getattr(simplefunctions, self.function_compare)(matrixes1)

            if fun_cmp > self.max_g[0]:
                self.clf_id[1] = self.clf_id[0]
                self.clf_id[0] = idx
                self.max_g[1] = self.max_g[0]
                self.max_g[0] = fun_cmp
            elif fun_cmp > self.max_g[1]:
                self.clf_id[2] = self.clf_id[1]
                self.clf_id[1] = idx
                self.max_g[2] = self.max_g[1]
                self.max_g[0] = fun_cmp
            elif fun_cmp > self.max_g[2]:
                self.clf_id[2] = idx
                self.max_g[2] = fun_cmp
        for clf_id in self.clf_id:
            if clf_id > len(self.estimators_ada) + len(self.estimators_bag):
                if clf_id % 2 == 0:
                    met = self.methods[0]
                    data_re, tar_re = met.fit_sample(X, y)
                    clf_ = clone(self.clfs[(clf_id-7)/2])
                    self.ensemble_.append(clf_.fit(data_re, tar_re))
                else:
                    met = self.methods[1]
                    data_re, tar_re = met.fit_sample(X, y)
                    clf_ = clone(self.clfs[(clf_id-7)/2])
                    self.ensemble_.append(clf_.fit(data_re, tar_re))
            else:
                clf_ = clone(self.clfs[clf_id])
                self.ensemble_.append(clf_.fit(X,y))

        meta_features = self._predict_meta_features(X)
        self.meta_clf_.fit(meta_features, y)

    def predict(self, X):
        meta_features = self._predict_meta_features(X)
        return self.meta_clf_.predict(meta_features)

    def _predict_meta_features(self, X):

        return np.column_stack([clf.predict(X) for clf in self.ensemble_])
