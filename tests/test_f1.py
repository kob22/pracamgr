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

# liczba testow f1
max_iter = 500

# klasyfikator
clf = tree.DecisionTreeClassifier()

# sprawdzian krzyzowy
folds = [StratifiedKFold(n_splits=10, random_state=5), KFold(n_splits=10, random_state=5)]
name_folds = ['Stratified CV, k=10', 'Unstratified CV, k=10']

# wykres
fig1 = plt.figure(facecolor='white', figsize=(7.532, 6))
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 10})
plt.rc('legend', fontsize=10)
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})


for id, (fold, name) in enumerate(zip(folds, name_folds)):
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(name)
    stdtpfp = []
    stdprere = []
    stdavg = []
    ftpfpall = []
    fprereall = []
    favgall = []

    # badanie zaleznosci miary f1 w stosunku do ilosci przykladow kl. mniejszosciowej
    for miniority in np.arange(1, 11, 1):

        max_items = miniority * 10

        # wybor 1000 obserwacji
        idx_maj = np.where(~mask)[0][:1000 - max_items]
        idx_min = np.where(mask)[0][:max_items]
        idx = np.concatenate((idx_min, idx_maj), axis=0)

        data, target = data_[idx, :], target_[idx]

        items = np.bincount(target)

        ftpfp = []
        fprere = []
        favg = []

        # powtarzanie iteracji
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

            # obliczanie f1
            f1 = f1_calculate(predictions, targets)
            ftpfp.append(f1[0])
            fprere.append(f1[1])
            favg.append(f1[2])

        # obliczanie std i sredniej
        stdtpfp.append(np.std(ftpfp))
        stdprere.append(np.std(fprere))
        stdavg.append(np.std(favg))

        ftpfpall.append(np.mean(ftpfp))
        fprereall.append(np.mean(fprere))
        favgall.append(np.mean(favg))
        print('----------------------------------')
        print('%s%% klasy mniejszosciowej' % miniority)
        print('')
        print('F1 TP FP = %s, od. std. = %s' % (np.mean(ftpfp), np.std(ftpfp)))
        print('F1 PR RE = %s, od. std. = %s' % (np.mean(fprere), np.std(fprere)))
        print('F1 AVG = %s, od. std. = %s' % (np.mean(favg), np.std(favg)))
        print('')

    # wyswietlanie wykresow
    if id == 0:
        ax1 = plt.subplot(2, 2, 1 + id)
    else:
        ax2 = plt.subplot(2, 2, 1 + id, sharey=ax1)
    plt.plot([x for x in range(1, 11)], stdtpfp, 's-', lw=2, label="F1 TP FP FN")
    plt.plot([x for x in range(1, 11)], stdprere, 'p-', lw=2, label="F1 PR RE")
    plt.plot([x for x in range(1, 11)], stdavg, '*-', lw=2, label="F1 AVG")
    if id == 0:
        plt.ylabel('Odchylenie standardowe')
    plt.xlabel('Procent klasy mniejszosciowej')
    plt.xlim(0.8, 10.2)
    plt.title('Od. std. F1, %s' % name)
    plt.legend(loc="upper right")

    if id == 0:
        ax3 = plt.subplot(2, 2, 3 + id, sharex=ax1)
    else:
        ax4 = plt.subplot(2, 2, 3 + id, sharex=ax2, sharey=ax3)
    plt.plot([x for x in range(1, 11)], ftpfpall, 's-', lw=2, label="F1 TP FP FN")
    plt.plot([x for x in range(1, 11)], fprereall, 'p-', lw=2, label="F1 PRE RE")
    plt.plot([x for x in range(1, 11)], favgall, '*-', lw=2, label="F1 AVG")
    if id == 0:
        plt.ylabel('Miara F1')
    plt.xlabel('Procent klasy mniejszosciowej')
    plt.xlim(0.8, 10.2)
    plt.title('Miara F1, %s' % name)
    plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(path, 'wyniki/wykresy_zdjecia/miara-F1.png'), dpi=120)
fig1.show()
raw_input()
