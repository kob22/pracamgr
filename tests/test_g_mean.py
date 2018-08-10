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
import os

path = os.path.dirname(os.path.abspath(__file__))

# generowanie danych
data___, target___ = make_classification(n_samples=1500, n_features=2, n_redundant=0, n_informative=2, n_classes=2,
                                         weights=[0.8, 0.2], random_state=5)
data__, target__ = make_imbalance(data___, target___, ratio=0.1, min_c_=1, random_state=23)
data_, target_ = shuffle(data__, target__)
mask = target_ == 1

# liczba testow g
max_iter = 500
# klasyfikator
clf = tree.DecisionTreeClassifier()
# sprawdzian krzyzowy
folds = [StratifiedKFold(n_splits=10, random_state=5), KFold(n_splits=10, random_state=5)]
name_folds = ['Strat. CV, k=10', 'Unstrat. CV, k=10']
# wykres
fig1 = plt.figure(facecolor='white', figsize=(7.532, 6))
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 10})
plt.rc('legend', fontsize=10)
for id, (fold, name) in enumerate(zip(folds, name_folds)):
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(name)
    stdtptn = []
    stdsesp = []
    stdavg = []
    gtptnall = []
    gsespall = []
    gavgall = []

    # badanie zaleznosci miary g w stosunku do ilosci przykladow kl. mniejszosciowej
    for miniority in np.arange(1, 11, 1):

        max_items = miniority * 10

        # wybor 1000 obserwacji
        idx_maj = np.where(~mask)[0][:1000 - max_items]
        idx_min = np.where(mask)[0][:max_items]
        idx = np.concatenate((idx_min, idx_maj), axis=0)

        data, target = data_[idx, :], target_[idx]

        items = np.bincount(target)

        gtptn = []
        gsesp = []
        gavg = []

        # powtarzanie testow
        for r in range(max_iter):
            # print(r)
            # klonowanie klasyfikatora
            clf_ = clone(clf)
            data_re, target_re = shuffle(data, target)
            skf = fold
            predictions = []
            targets = []

            # sprawdzian krzyzowy
            for train_index, test_index in skf.split(data_re, target_re):
                clf_train_ = clone(clf)

                clf_train_.fit(data_re[train_index], target_re[train_index])
                predictions.append(clf_train_.predict(data_re[test_index]))
                targets.append(target_re[test_index])

            # obliczanie g
            g = g_calculate(predictions, targets)

            gtptn.append(g[0])
            gsesp.append(g[1])
            gavg.append(g[2])

        # obliczanie wynikow
        stdtptn.append(np.std(gtptn))
        stdsesp.append(np.std(gsesp))
        stdavg.append(np.std(gavg))

        gtptnall.append(np.mean(gtptn))
        gsespall.append(np.mean(gsesp))
        gavgall.append(np.mean(gavg))
        print('----------------------------------')
        print('%s%% klasy mniejszosciowej' % miniority)
        print('')
        print('G-mean TP FP = %s, od. std. = %s' % (np.mean(gtptn), np.std(gtptn)))
        print('G-mean Se Sp = %s, od. std. = %s' % (np.mean(gsesp), np.std(gsesp)))
        print('G-mean AVG = %s, od. std. = %s' % (np.mean(gavg), np.std(gavg)))
        print('')

    # wyswietlanie wykresow
    if id == 0:
        ax1 = plt.subplot(2, 2, 1 + id)
    else:
        ax2 = plt.subplot(2, 2, 1 + id, sharey=ax1)
    plt.plot([x for x in range(1, 11)], stdtptn, 's-', lw=2, label="G-mean TP FN ")
    plt.plot([x for x in range(1, 11)], stdsesp, 'p-', lw=2, label="G-mean Se Sp")
    plt.plot([x for x in range(1, 11)], stdavg, '*-', lw=2, label="G-mean AVG")
    if id == 0:
        plt.ylabel('Odchylenie standardowe')
    plt.xlabel('Procent klasy mniejszosciowej')
    plt.xlim(0, 11)
    plt.xlim(0.8, 10.2)
    plt.title('Od. std. G-mean, %s' % name)
    plt.legend(loc="upper right")

    if id == 0:
        ax3 = plt.subplot(2, 2, 3 + id, sharex=ax1)
    else:
        ax4 = plt.subplot(2, 2, 3 + id, sharex=ax2, sharey=ax3)
    plt.plot([x for x in range(1, 11)], gtptnall, 's-', lw=2, label="G-mean TP FN ")
    plt.plot([x for x in range(1, 11)], gsespall, 'p-', lw=2, label="G-mean Se Sp")
    plt.plot([x for x in range(1, 11)], gavgall, '*-', lw=2, label="G-mean AVG")
    if id == 0:
        plt.ylabel('Miara G-mean')

    plt.xlabel('Procent klasy mniejszosciowej')
    plt.xlim(0.8, 10.2)
    plt.title('Miara G-mean, %s' % name)
    plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(path, 'wyniki/wykresy_zdjecia/miara-G-mean.png'), dpi=120)
fig1.show()
raw_input()

