from sklearn.datasets import make_classification, make_moons, make_gaussian_quantiles
from sklearn import tree
from cross_val.cross_val import cross_val_pred2ict
from simplefunctions import *
from imblearn.datasets import make_imbalance
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC, SVC
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from data import importdata
from scipy import interp
import os

path = os.path.dirname(os.path.abspath(__file__))

dataset = ['load_transfusion']
db = getattr(importdata, dataset[0])()
data_re = db.data
target_re = db.target

# klasyfikator
clf = GaussianNB()
# CV
folds = [StratifiedKFold(n_splits=3, random_state=5), KFold(n_splits=3, random_state=5)]
name_folds = ['Stratified K-fold, k=3', 'Unstratified K-fold, k=3']

# wykres
fig1 = plt.figure(facecolor='white', figsize=(6, 9))
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 10})
plt.rc('legend', fontsize=10)
for id, (fold, name) in enumerate(zip(folds, name_folds)):

    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(name)

    skf = fold
    print(skf)
    predictions = []
    targets = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    cross = list(skf.split(data_re, target_re))
    ax = plt.subplot(2, 1, 1 + id)
    # sprawdzian krzyzowy
    for cv, (train_index, test_index) in enumerate(cross):
        clf_train_ = clone(clf)

        probas = clf_train_.fit(data_re[train_index], target_re[train_index]).predict_proba(data_re[test_index])
        # obliczanie krzywej ROC
        fpr, tpr, thresholds = roc_curve(target_re[test_index],
                                         probas[:, 1],
                                         pos_label=1)

        # srednia ROC
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        # wykres ROC FOLD
        plt.plot(fpr,
                 tpr,
                 lw=2,
                 label='ROC fold %d (area = %0.2f)'
                       % (cv + 1, roc_auc))
        predictions.extend(probas[:, 1])
        targets.extend(target_re[test_index])

    mean_tpr /= len(cross)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # wykres
    plt.title('ROC, %s' % name)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='ROC AVG (area = %0.2f)' % mean_auc, lw=3)

    fpr, tpr, thresholds = roc_curve(targets,
                                     predictions,
                                     pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC merge (area = %0.2f)' % roc_auc, lw=3)
    plt.legend(loc="lower right")

plt.savefig(os.path.join(path, 'wyniki/wykresy_zdjecia/CV_ROC.png'))
fig1.show()
raw_input()
