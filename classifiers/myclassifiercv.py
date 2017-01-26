import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from simplefunctions import print_scores, confusion_matrix, f1tpfp
from sklearn.model_selection import StratifiedKFold
from cross_val.cross_val import cross_val_pred2ict


def _parallel_fit_estimator(estimator, X, y):
    """Private function used to fit an estimator within a job."""
    estimator.fit(X, y)
    return estimator


class F1ClassifierCV(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, estimators, n_jobs=1, n_folds=10, random_state=None):
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.n_jobs = n_jobs
        self.experts = [[], []]
        self.max_f1 = [-1, -1]
        self.groups = []
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, y):

        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        skf = StratifiedKFold(n_splits=self.n_folds, random_state=self.random_state)

        self.estimators_ = [clone(estimator) for _, estimator in self.estimators]

        cv_predictions = []
        targets = []

        for estimator in self.estimators_:
            testpredict, testtarget = cross_val_pred2ict(estimator, X, y, cv=skf.get_n_splits(X, y),
                                                         n_jobs=-1)
            cv_predictions.append((testpredict))
            targets.append(testtarget)

        self.groups = np.unique(y)

        for idx, (prediction, target) in enumerate(zip(cv_predictions, targets)):
            matrixes1 = []
            matrixes2 = []
            for pred, tar in zip(prediction, target):
                matrixes1.append(confusion_matrix(tar, pred))
            class1 = precision(matrixes1)[0]
            if class1 > self.max_f1[0]:
                self.max_f1[0] = class1
                self.experts[0] = idx
            elif class1 == self.max_f1[0]:
                self.experts[0].append(idx)

            for matrix in matrixes1:
                matrixes2.append(np.array([[matrix[1, 1], matrix[1, 0]], [matrix[0, 1], matrix[0, 0]]]))

            class2 = precision(matrixes2)[0]
            if class2 > self.max_f1[1]:
                self.max_f1[1] = class2
                self.experts[1] = idx
            elif class1 == self.max_f1[1]:
                self.experts[1].append(idx)
                # print(print_scores(prediction,target))
        print(self.experts)
        print(self.max_f1)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_estimator)(clone(clf), X, y)
            for _, clf in self.estimators)
        return self

    def predict(self, X):

        predictions = self._predict(X)
        final_predictions = []
        for line in predictions:
            size = np.unique(line).size
            if size == 1:
                final_predictions.append(line[0])
            elif size > 1:
                if line[self.experts[0]] == self.groups[0]:
                    if line[self.experts[1]] == self.groups[1]:
                        if self.max_f1[0] > self.max_f1[1]:
                            final_predictions.append(self.groups[0])
                        elif self.max_f1[0] < self.max_f1[1]:
                            final_predictions.append(self.groups[1])
                        else:
                            list = [0, 1, 2]
                            for expert in self.experts:
                                list.remove(expert)
                            final_predictions.append(line[list])
                    else:
                        final_predictions.append(self.groups[0])
                elif line[self.experts[1]] == self.groups[1]:
                    final_predictions.append(self.groups[1])
                else:
                    final_predictions.append(np.argmax(np.bincount(line)))

        # maj = np.apply_along_axis(lambda x:
        #                           np.argmax(np.bincount(x)),
        #                           axis=1,
        #                           arr=predictions.astype('int'))



        return final_predictions

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T
