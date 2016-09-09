from __future__ import division
import math
from sklearn.utils.multiclass  import unique_labels
from sklearn.metrics.classification import _check_targets
from scipy.sparse import coo_matrix
import numpy as np

def avarage_score(scores):
    e = float(sum(scores)) / len(scores)

    suma = 0
    for score in scores:
        suma += (score - e) ** 2

    s = math.sqrt(suma/(len(scores)-1))
    return round(e,3),round(s,3)

def avg(scores):
    e = float(sum(scores)) / len(scores)

    return e



def accuracy(matrixs):
    avg_acc = []
    sum_tptn = 0
    sum_all = 0
    for matrix in matrixs:
        tptn = matrix[0,0]+matrix[1,1]
        all = sum(y for y in (sum(x for x in matrix)))

        sum_tptn += tptn
        sum_all += all

        avg_acc.append((float(tptn)/all))

    return avg(avg_acc), float(sum_tptn)/sum_all

def precision(matrixs):
    avg_ppv = []
    sum_tp = 0
    sum_tpfp = 0

    for matrix in matrixs:
        tp = matrix[0,0]
        tpfp = matrix[0,0] + matrix[1,0]

        sum_tp += tp
        sum_tpfp += tpfp

        avg_ppv.append(float(tp)/tpfp)
    print(avg_ppv)
    return avg(avg_ppv), float(sum_tp)/sum_tpfp


def sensitivity(matrixs):
    avg_tpr = []
    sum_tp = 0
    sum_tpfn = 0

    for matrix in matrixs:
        tp = matrix[0,0]
        tpfn = matrix[0,0] + matrix[0,1]

        sum_tp += tp
        sum_tpfn += tpfn

        avg_tpr.append(float(tp)/tpfn)
    print(avg_tpr)
    return avg(avg_tpr), float(sum_tp)/sum_tpfn

def f1tpfp(matrixs):
    avg_ppv = []
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
    for matrix in matrixs:
        sum_tp += matrix[0,0]
        sum_fp += matrix[1,0]
        sum_fn += matrix[0,1]


    return float(2*sum_tp)/(2*sum_tp+sum_fp+sum_fn)

def f1prre(pre,rec):

    return 2*(pre*rec)/float(pre+rec)


def f1avg(matrixs):
    avg_f1 = []


    for matrix in matrixs:
        tp = matrix[0,0]
        fp = matrix[1,0]
        fn = matrix[0,1]


        avg_f1.append(float(2*tp)/(2*tp+fp+fn))
    print("f1")
    print(avg_f1)
    return avg(avg_f1)

def confusion_matrix(y_true, y_pred, labels=None):
    """Compute confusion matrix to evaluate the accuracy of a classification

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.

    Read more in the :ref:`User Guide <confusion_matrix>`.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.

    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    Returns
    -------
    C : array, shape = [n_classes, n_classes]
        Confusion matrix

    References
    ----------
    .. [1] `Wikipedia entry for the Confusion matrix
           <http://en.wikipedia.org/wiki/Confusion_matrix>`_

    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    n_labels = labels.size

    label_to_ind = dict((y, x) for x, y in enumerate(labels))
    # convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]


    if n_labels == 1:
        n_labels = 2
    CM = coo_matrix((np.ones(y_true.shape[0], dtype=np.int), (y_true, y_pred)),
                    shape=(n_labels, n_labels)
                    ).toarray()
    return CM