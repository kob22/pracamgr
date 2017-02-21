import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
import simplefunctions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from classifiers.clf_expert import ensembel_rating
from cross_val.cross_val import cross_val_pred2ict
from classifiers.stacking import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def _parallel_fit_estimator(estimator, X, y):
    """Private function used to fit an estimator within a job."""
    estimator.fit(X, y)
    return estimator


# klasyfikator ENSEBLE z funckja wybierajaca najlepszy klasyfikator na podstawie sprawdzianu krzyzowego
# funkcja porownujaca klasyfikatory moze byc: precision_tp_fp, g_meantpfp, f1tpfp
class meta_classifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    # inicjalizacja

    # function_compare - funkcja porownujaca klasyfikatory
    def __init__(self, estimators, estimators_bag, estimators_ada, n_jobs=1, function_compare='precision_tp_fp',
                 n_folds=3, n_estimators=100):
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.n_jobs = n_jobs
        self.experts = [-1, -1]
        self.max_rating = [-1, -1]
        self.groups = []
        self.g_mean = [-1, -1]
        self.function_compare = function_compare
        self.clfs = []
        self.clfs_ensemble = []
        self.n_folds = n_folds
        self.max_g = [-1, -1]
        self.clf_id = [-1, -1]
        self.n_estimators = n_estimators
        self.meta_clf_ = MLPClassifier(solver='lbfgs', random_state=1)
        self.clfs_ensemble = []
        self.estimators_bag = estimators_bag
        self.estimators_ada = estimators_ada

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

        # ocena klasyfikatorow
        for clf in self.clfs:
            testpredict, testtarget = cross_val_pred2ict(clf, X, y, cv=self.n_folds,
                                                         n_jobs=1)
            cv_predictions.append((testpredict))
            targets.append(testtarget)

        # wylanianie 2 najlepszych ekspertow
        for idx, (prediction, target) in enumerate(zip(cv_predictions, targets)):

            matrixes1 = []
            matrixes2 = []
            for pred, tar in zip(prediction, target):
                matrixes1.append(simplefunctions.confusion_matrix(tar, pred))
            for matrix in matrixes1:
                matrixes2.append(np.array([[matrix[1, 1], matrix[1, 0]], [matrix[0, 1], matrix[0, 0]]]))
            g_mean = simplefunctions.precision_tp_fp(matrixes1)

            if g_mean > self.max_g[0]:
                self.clf_id[1] = self.clf_id[0]
                self.clf_id[0] = idx
                self.max_g[1] = self.max_g[0]
                self.max_g[0] = g_mean
            elif g_mean > self.max_g[1]:
                self.clf_id[1] = idx
                self.max_g[1] = g_mean

        self.clfs_ensemble.append(ensembel_rating(self.estimators))
        for clf in self.clf_id:
            self.clfs_ensemble.append(self.clfs[clf])

        self.ensemble_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_estimator)(clone(clf), X, y)
            for clf in self.clfs_ensemble)

        meta_features = self._predict_meta_features(X)
        self.meta_clf_.fit(meta_features, y)

    def predict(self, X):
        meta_features = self._predict_meta_features(X)
        return self.meta_clf_.predict(meta_features)

    def _predict_meta_features(self, X):

        return np.column_stack([clf.predict(X) for clf in self.ensemble_])
