import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
import simplefunctions


def _parallel_fit_estimator(estimator, X, y):
    """Private function used to fit an estimator within a job."""
    estimator.fit(X, y)
    return estimator


# klasyfikator ekspercki
# funkcja porownujaca klasyfikatory moze byc: precision_tp_fp, g_meantpfp, f1tpfp
class clf_expert(BaseEstimator, ClassifierMixin, TransformerMixin):
    # inicjalizacja

    # function_compare - funkcja porownujaca klasyfikatory
    def __init__(self, estimators, n_jobs=1, function_compare='precision_tp_fp'):
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.n_jobs = n_jobs
        self.experts = [-1, -1]
        self.max_rating = [-1, -1]
        self.groups = []
        self.g_mean = [-1, -1]
        self.function_compare = function_compare

    def fit(self, X, y):

        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')
        # klonowanie i trenowanie
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_estimator)(clone(clf), X, y)
            for _, clf in self.estimators)


        self.groups = np.unique(y)

        # wylanianie ekspertow w swojej klasie
        for idx, clf in enumerate(self.estimators_):

            matrix1 = [np.asarray(simplefunctions.confusion_matrix(y, clf.predict(X)))]
            class1 = getattr(simplefunctions, self.function_compare)(matrix1)

            if class1 > self.max_rating[0]:
                self.max_rating[0] = class1
                self.experts[0] = (idx)
                self.g_mean[0] = simplefunctions.g_meantpfp(matrix1)
            elif class1 == self.max_rating[0]:
                self.experts[0] = (idx)
                self.g_mean[0] = simplefunctions.g_meantpfp(matrix1)

            matrix2 = [np.array([[matrix1[0][1, 1], matrix1[0][1, 0]], [matrix1[0][0, 1], matrix1[0][0, 0]]])]
            class2 = getattr(simplefunctions, self.function_compare)(matrix2)
            if class2 > self.max_rating[1]:
                self.max_rating[1] = class2
                self.experts[1] = idx
                self.g_mean[1] = simplefunctions.g_meantpfp(matrix1)
            elif class1 == self.max_rating[1]:
                self.experts[1] = idx
                self.g_mean[1] = simplefunctions.g_meantpfp(matrix1)

        return self

    # klasyfikacja
    def predict(self, X):

        predictions = self._predict(X)
        final_predictions = []
        for id, line in enumerate(predictions):
            size = np.unique(line).size
            # jezeli jest zgodnosc co do klasy nadaje wskazana klase
            if size == 1:
                final_predictions.append(line[0])
            elif size > 1:
                # sprawdza czy jest eksperci wskazali swoja klase, jezeli tylko 1 ekspert wskazal swoja klase to wybierana jest ona
                # jezeli 2 ekspertow wskaze swoje klasy, wybierana jest ta z wiekszym prawdopodobnienstwem
                # jezeli prawdopodobienstwo jest jednakowe, wybierany jest klasyfikator z wiekszym g-mean
                if line[self.experts[0]] == self.groups[0]:
                    if line[self.experts[1]] == self.groups[1]:

                        proba = [self._clf_predict_proba(self.experts[0], X[id].reshape(1, -1))[0][0],
                                 self._clf_predict_proba(self.experts[1], X[id].reshape(1, -1))[0][1]]
                        if proba[0] > proba[1]:
                            final_predictions.append(self.groups[0])
                        elif proba[1] > proba[0]:
                            final_predictions.append(self.groups[1])
                        else:
                            if self.g_mean[0] > self.g_mean[1]:
                                final_predictions.append(self.groups[0])
                            else:
                                final_predictions.append(self.groups[0])

                    else:
                        final_predictions.append(self.groups[0])
                elif line[self.experts[1]] == self.groups[1]:
                    final_predictions.append(self.groups[1])
                else:
                    final_predictions.append(np.argmax(np.bincount(line)))

        return final_predictions

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T

    def _predict_proba(self, X):
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _clf_predict_proba(self, clf_id, X):
        return self.estimators_[clf_id].predict_proba(X)
