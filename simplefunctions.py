from __future__ import division
import math
from sklearn.utils.multiclass  import unique_labels
from sklearn.metrics.classification import _check_targets
from scipy.sparse import coo_matrix
import numpy as np
import warnings
from texttable import Texttable

def avarage_score(scores):
    if len(scores) > 0:
        e = float(sum(scores)) / len(scores)

        suma = 0
        for score in scores:
            suma += (score - e) ** 2

        s = math.sqrt(suma / (len(scores) - 1))
        return round(e, 3), round(s, 3)
    else:
        warnings.warn("No values", stacklevel=2)
        return 0

def avg(scores):
    if len(scores) > 0:
        e = float(sum(scores)) / len(scores)

        return e
    else:
        warnings.warn("No values", stacklevel=2)
    return 0


def accuracy(matrixs):
    avg_acc = []
    sum_tptn = 0
    sum_all = 0

    for matrix in matrixs:
        tptn = matrix[0,0]+matrix[1,1]
        all = sum(y for y in (sum(x for x in matrix)))

        sum_tptn += tptn
        sum_all += all
        if all != 0:
            avg_acc.append((float(tptn) / all))
        else:
            warnings.warn("Warning, Accuracy 0 - no samples", stacklevel=2)

    if sum_all != 0:
        return avg(avg_acc), float(sum_tptn) / sum_all
    else:
        warnings.warn("Warning, Accuracy 0 - no samples", stacklevel=2)
    return avg(avg_acc), 0


def tpr(matrixs):
    sum_tp = 0
    sum_tpfn = 0

    for matrix in matrixs:
        tp = matrix[0, 0]
        tpfn = matrix[0, 0] + matrix[0, 1]

        sum_tp += tp
        sum_tpfn += tpfn

    if sum_tpfn != 0:
        return float(sum_tp) / sum_tpfn
    else:
        warnings.warn("Warning, TPR = 0 - no predicted samples", stacklevel=2)
        return 0


def tnr(matrixs):
    sum_tn = 0
    sum_tnfp = 0

    for matrix in matrixs:
        tn = matrix[1, 1]
        tnfp = matrix[1, 1] + matrix[1, 0]

        sum_tn += tn
        sum_tnfp += tnfp

    if sum_tnfp != 0:
        return float(sum_tn) / sum_tnfp
    else:
        warnings.warn("Warning, TNR = 0 - no predicted samples", stacklevel=2)
        return 0


def specificity_tn_fp(matrixs):
    sum_tn = 0
    sum_tnfp = 0

    for matrix in matrixs:
        tn = matrix[1, 1]
        tnfp = matrix[1, 1] + matrix[1, 0]

        sum_tn += tn
        sum_tnfp += tnfp

    if sum_tnfp != 0:
        return float(sum_tn) / sum_tnfp
    else:
        warnings.warn("Warning, Specifity = 0 - no predicted samples", stacklevel=2)
        return 0


def specificity_avg(matrixs):
    avg_specifity = []

    for matrix in matrixs:
        tp = matrix[1, 1]
        tnfp = matrix[1, 1] + matrix[1, 0]

    if tnfp != 0:
        avg_specifity.append(float(tp) / tnfp)
    else:
        warnings.warn("Warning, Specifity = 0 - no predicted samples", stacklevel=2)
        avg_specifity.append(0)
    return np.mean(avg_specifity), np.std(avg_specifity)


def specificities(matrixs):
    avg_specifity = []

    for matrix in matrixs:
        tp = matrix[1, 1]
        tnfp = matrix[1, 1] + matrix[1, 0]

    if tnfp != 0:
        avg_specifity.append(float(tp) / tnfp)
    else:
        warnings.warn("Warning, Specifity = 0 - no predicted samples", stacklevel=2)
        return 0
    return avg_specifity


# liczy precyzje z tp i fp
def precision_tp_fp(matrixs):
    sum_tp = 0
    sum_tpfp = 0

    for matrix in matrixs:
        tp = matrix[0,0]
        tpfp = matrix[0,0] + matrix[1,0]

        sum_tp += tp
        sum_tpfp += tpfp

    if sum_tpfp != 0:
        return float(sum_tp) / sum_tpfp
    else:
        warnings.warn("Warning, Precision = 0 - no predicted samples", stacklevel=2)
        return 0


# liczy precyzje z kross walidacji, zwraca srednia precyja + odchylenie
def precision_avg(matrixs):
    avg_ppv = []

    for matrix in matrixs:
        tp = matrix[0, 0]
        tpfp = matrix[0, 0] + matrix[1, 0]

        if tpfp != 0:
            avg_ppv.append(float(tp) / tpfp)
        else:
            warnings.warn("Warning, Precision = 0 - no predicted samples", stacklevel=2)
            avg_ppv.append(0)

    return np.mean(avg_ppv), np.std(avg_ppv)


#liczy precyzje z kross walidacji, zwraca tablice precyzji
def precisions(matrixs):
    avg_ppv = []


    for matrix in matrixs:
        tp = matrix[0, 0]
        tpfp = matrix[0, 0] + matrix[1, 0]


        if tpfp != 0:
            avg_ppv.append(float(tp) / tpfp)
        else:
            warnings.warn("Warning, Precision = 0 - no predicted samples", stacklevel=2)
            avg_ppv.append(0)

    return avg_ppv


# liczy czulosc z tp fp
def sensitivity_tp_fp(matrixs):
    sum_tp = 0
    sum_tpfn = 0

    for matrix in matrixs:
        tp = matrix[0,0]
        tpfn = matrix[0,0] + matrix[0,1]

        sum_tp += tp
        sum_tpfn += tpfn

    if sum_tpfn != 0:
        return float(sum_tp) / sum_tpfn
    else:
        warnings.warn("Warning, Sensitivity = 0 - no predicted samples", stacklevel=2)
        return float(sum_tp) / sum_tpfn


# liczy srednia czulosc z kross walidacji
def sensitivity_avg(matrixs):
    avg_tpr = []

    for matrix in matrixs:
        tp = matrix[0, 0]
        tpfn = matrix[0, 0] + matrix[0,1]

        if tpfn != 0:
            avg_tpr.append(float(tp) / tpfn)
        else:
            warnings.warn("Warning, Sensitivity = 0 - no predicted samples", stacklevel=2)
            avg_tpr.append(0)

    return np.mean(avg_tpr), np.std(avg_tpr)


#liczy czulosc, zwraca tablie czulosci z kross walidacji
def sensitivities(matrixs):
    avg_tpr = []

    for matrix in matrixs:
        tp = matrix[0, 0]
        tpfn = matrix[0, 0] + matrix[0, 1]

        if tpfn != 0:
            avg_tpr.append(float(tp) / tpfn)
        else:
            warnings.warn("Warning, Sensitivity = 0 - no predicted samples", stacklevel=2)
            avg_tpr.append(0)

    return avg_tpr


def f1prre(precisions, recalls):
    guard = 0
    if 0 in precisions:
        warnings.warn("At least one fold has Precision = 0, F1 can be calculated wrong", stacklevel=2)
        guard += 1
    if 0 in recalls:
        warnings.warn("At least one fold has Recall = 0, F1 can be calculated wrong", stacklevel=2)
        guard += 1

    if guard == 2:
        return 0
    else:
        avgprecision = avg(precisions)
        avgrecall = avg(recalls)
        return 2 * (avgprecision * avgrecall) / float(avgprecision + avgrecall)


def f1tpfp(matrixs):

    sum_tp = 0
    sum_fp = 0
    sum_fn = 0

    for matrix in matrixs:
        sum_tp += matrix[0,0]
        sum_fp += matrix[1,0]
        sum_fn += matrix[0,1]

    if sum_tp == 0:
        warnings.warn("Warning, F1 = 0 - no predicted samples", stacklevel=2)
        return 0
    return float(2*sum_tp)/(2*sum_tp+sum_fp+sum_fn)




def f1avg(matrixs):
    avg_f1 = []


    for matrix in matrixs:
        tp = matrix[0,0]
        fp = matrix[1,0]
        fn = matrix[0,1]


        avg_f1.append(float(2*tp)/(2*tp+fp+fn))
    if 0 in avg_f1:
        warnings.warn("At least one fold has F1 = 0, F1 can be calculated wrong", stacklevel=2)

    # print(avg_f1)
    return avg(avg_f1)


def g_mean(sensitivity, specificity):
    return math.sqrt((sensitivity * specificity))


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

def print_matrix(matrixes):
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for matrix in matrixes:
        tp += matrix[0][0]
        fn += matrix[0][1]
        fp += matrix[1][0]
        tn += matrix[1][1]
        print("---\n|%d %d|\n|%d %d|\n" % (matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]))

    print("Suma ")
    print("---\n|%d %d|\n|%d %d|\n" % (tp, fn, fp, tn))


def print_complete_matrix(matrixes, groups):
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for matrix in matrixes:
        tp += matrix[0][0]
        fn += matrix[0][1]
        fp += matrix[1][0]
        tn += matrix[1][1]
    table = Texttable()
    table.add_rows([groups, [tp, fn], [fp, tn]])
    print(table.draw())


def print_to_latex(predict, target):
    matrices0 = []
    matrices1 = []

    if len(predict) != len(target):
        raise ValueError('length score and target are different!')
    for pr, tar in zip(predict, target):
        matrices0.append(confusion_matrix(tar, pr))

    acc = accuracy(matrices0)
    tprate = tpr(matrices0)
    tnrate = tnr(matrices0)

    prec0 = precision_tp_fp(matrices0)
    sens0 = sensitivity_tp_fp(matrices0)
    spec0 = specificity_tn_fp(matrices0)
    g0 = g_mean(sens0, spec0)
    for matrix in matrices0:
        matrices1.append(np.array([[matrix[1, 1], matrix[1, 0]], [matrix[0, 1], matrix[0, 0]]]))

    prec1 = precision_tp_fp(matrices1)
    sens1 = sensitivity_tp_fp(matrices1)
    spec1 = specificity_tn_fp(matrices1)
    g1 = g_mean(sens1, spec1)
    f1 = f1tpfp(matrices1)
    return float("{0:.2f}".format(acc[1])), float("{0:.2f}".format(tnrate)), float("{0:.2f}".format(g0)), float(
        "{0:.2f}".format(f1))




def print_scores(predict, target):
    matrices0 = []
    matrices1 = []

    if len(predict) != len(target):
        raise ValueError('length score and target are different!')
    for pr, tar in zip(predict, target):
        matrices0.append(confusion_matrix(tar, pr))
    allgroups = []
    for fold in target:
        allgroups.extend(np.unique(fold))

    groups = np.unique(allgroups)

    print_complete_matrix(matrices0, groups)
    print("Accuracy: %r" % str(accuracy(matrices0)))
    cols_name = ['', 'Precision', 'Sensitivity', 'Specificity', 'F1tpfp', 'F1prre', 'F1AVG', 'G-mean']
    prec = precision_tp_fp(matrices0)
    sens = sensitivity_tp_fp(matrices0)
    spec = specificity_tn_fp(matrices0)
    firstparams = [groups[0], prec, sens, spec, f1tpfp(matrices0),
                   f1prre(precisions(matrices0), sensitivities(matrices0)),
                   f1avg(matrices0), g_mean(sens, spec)]

    for matrix in matrices0:
        matrices1.append(np.array([[matrix[1, 1], matrix[1, 0]], [matrix[0, 1], matrix[0, 0]]]))

    prec = precision_tp_fp(matrices1)
    sens = sensitivity_tp_fp(matrices1)
    spec = specificity_tn_fp(matrices1)

    secondparams = [groups[1], prec, sens, spec, f1tpfp(matrices1),
                    f1prre(precisions(matrices1), sensitivities(matrices1)), f1avg(matrices1), g_mean(sens, spec)]

    table = Texttable()
    table.add_rows([cols_name, firstparams, secondparams])
    print(table.draw())
    # poprawic warningi w precision itd
    #sprawdzi f1, gdy precision jest 0
