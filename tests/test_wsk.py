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
from sklearn.naive_bayes import GaussianNB

path = os.path.dirname(os.path.abspath(__file__))

# generowanie danych
data___, target___ = make_classification(n_samples=1500, n_features=2, n_redundant=0, n_informative=2, n_classes=2,
                                         weights=[0.8, 0.2], random_state=5)
data__, target__ = make_imbalance(data___, target___, ratio=0.1, min_c_=1, random_state=23)
data_, target_ = shuffle(data__, target__)
mask = target_ == 1

# liczba testow
max_iter = 500
# klasyfikator
clf = tree.DecisionTreeClassifier()

# deklaracja CV
folds = [StratifiedKFold(n_splits=10, random_state=5), KFold(n_splits=10, random_state=5)]
name_folds = ['Stratified K-fold, k=10', 'Unstratified K-fold, k=10']

# wykresy
fig1 = plt.figure(facecolor='white', figsize=(8.232, 10))
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 10})
plt.rc('legend', fontsize=10)
fig1 = plt.figure(facecolor='white', figsize=(8.232, 10))
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 10})
plt.rc('legend', fontsize=10)


for id, (fold, name) in enumerate(zip(folds, name_folds)):
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(name)
    stdsens = []
    stdspec = []
    stdprec = []

    avgsens = []
    avgspec = []
    avgprec = []

    # badanie zaleznosci miar w stosunku do ilosci przykladow kl. mniejszosciowej
    for miniority in np.arange(1, 11, 1):

        max_items = miniority * 10

        # wybor 1000 obserwacji
        idx_maj = np.where(~mask)[0][:1000 - max_items]
        idx_min = np.where(mask)[0][:max_items]
        idx = np.concatenate((idx_min, idx_maj), axis=0)

        data, target = data_[idx, :], target_[idx]

        items = np.bincount(target)

        sens = []
        spec = []
        prec = []

        # powtarzanie wieloktorne na tych samych danych
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

            # print(predictions)


            sens.extend(sensivity_calc(predictions, targets))
            spec.extend(specif_calc(predictions, targets))
            prec.extend(precion_calc(predictions, targets))

        # dodawanie wynikow i obliczanie STD i sredniej
        stdsens.extend([np.std(sens[::2]), np.std(sens[1::2])])
        stdspec.extend([np.std(spec[::2]), np.std(spec[1::2])])
        stdprec.extend([np.std(prec[::2]), np.std(prec[1::2])])

        avgsens.extend([np.mean(sens[::2]), np.mean(sens[1::2])])
        avgspec.extend([np.mean(spec[::2]), np.mean(spec[1::2])])
        avgprec.extend([np.mean(prec[::2]), np.mean(prec[1::2])])
        print('----------------------------------')
        print('%s%% klasy mniejszosciowej' % miniority)
        print('')
        print('Sensitivity TP FP = %s, od. std. = %s' % (np.mean(sens[::2]), np.std(sens[::2])))
        print('sensitivity AVG = %s, od. std. = %s' % (np.mean(sens[1::2]), np.std(sens[1::2])))
        print('Specificity TP FP = %s, od. std. = %s' % (np.mean(spec[::2]), np.std(spec[::2])))
        print('Specificity AVG = %s, od. std. = %s' % (np.mean(spec[1::2]), np.std(spec[1::2])))
        print('Precision TP FP = %s, od. std. = %s' % (np.mean(prec[::2]), np.std(prec[::2])))
        print('Precision AVG = %s, od. std. = %s' % (np.mean(prec[1::2]), np.std(prec[1::2])))
        print('')

    # wyswietlanie wykresow
    plt.figure(1)
    if id == 0:
        ax1 = plt.subplot(3, 2, 1 + id)
    else:
        ax2 = plt.subplot(3, 2, 1 + id, sharey=ax1)

    plt.title('%s' % name)
    plt.plot([x for x in range(1, 11)], stdsens[::2], 's-', lw=2, label="Sensitivity TP FN ")
    plt.plot([x for x in range(1, 11)], stdsens[1::2], 'p-', lw=2, label="Sensitivity AVG")
    if id == 0:
        plt.ylabel('Odchylenie standardowe')

    plt.xlim(0, 11)

    if id == 0:
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                          box.width, box.height * 0.8])

        # Put a legend below current axis
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                   fancybox=True, shadow=True)
    else:
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0 + box.height * 0.2,
                          box.width, box.height * 0.8])

        # Put a legend below current axis
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                   fancybox=True, shadow=True)
    if id == 0:
        ax3 = plt.subplot(3, 2, 3 + id)
    else:
        ax4 = plt.subplot(3, 2, 3 + id, sharey=ax3)
    plt.plot([x for x in range(1, 11)], stdspec[::2], 's-', lw=2, label="Specificity TP FN ")
    plt.plot([x for x in range(1, 11)], stdspec[1::2], 'p-', lw=2, label="Specificity AVG")
    if id == 0:
        plt.ylabel('Odchylenie standardowe')

    plt.xlim(0, 11)
    if id == 0:
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0 + box.height * 0.2,
                          box.width, box.height * 0.8])

        # Put a legend below current axis
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                   fancybox=True, shadow=True)

    else:
        box = ax4.get_position()
        ax4.set_position([box.x0, box.y0 + box.height * 0.2,
                          box.width, box.height * 0.8])

        # Put a legend below current axis
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                   fancybox=True, shadow=True)
    if id == 0:
        ax5 = plt.subplot(3, 2, 5 + id)
    else:
        ax6 = plt.subplot(3, 2, 5 + id, sharey=ax5)

    plt.plot([x for x in range(1, 11)], stdprec[::2], 's-', lw=2, label="Precision TP FN ")
    plt.plot([x for x in range(1, 11)], stdprec[1::2], 'p-', lw=2, label="Precision AVG")
    if id == 0:
        plt.ylabel('Odchylenie standardowe')
    plt.xlabel('Procent klasy mniejszosciowej')
    plt.xlim(0, 11)

    if id == 0:
        box = ax5.get_position()
        ax5.set_position([box.x0, box.y0 + box.height * 0.1,
                          box.width, box.height * 0.9])

        # Put a legend below current axis
        ax5.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
                   fancybox=True, shadow=True)
    else:
        box = ax6.get_position()
        ax6.set_position([box.x0, box.y0 + box.height * 0.1,
                          box.width, box.height * 0.9])

        # Put a legend below current axis
        ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
                   fancybox=True, shadow=True)

    plt.figure(2)
    if id == 0:
        ax21 = plt.subplot(3, 2, 1 + id)
    else:
        ax22 = plt.subplot(3, 2, 1 + id, sharey=ax21)
    plt.title('%s' % name)
    plt.plot([x for x in range(1, 11)], avgsens[::2], 's-', lw=2, label="Sensitivity TP FN ")
    plt.plot([x for x in range(1, 11)], avgsens[1::2], 'p-', lw=2, label="Sensitivity AVG")
    if id == 0:
        plt.ylabel('Sensitivity')

    plt.xlim(0, 11)
    if id == 0:
        box = ax21.get_position()
        ax21.set_position([box.x0, box.y0 + box.height * 0.2,
                           box.width, box.height * 0.8])

        # Put a legend below current ax2is
        ax21.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                    fancybox=True, shadow=True)

    else:

        box = ax22.get_position()
        ax22.set_position([box.x0, box.y0 + box.height * 0.2,
                           box.width, box.height * 0.8])

        # Put a legend below current ax2is
        ax22.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                    fancybox=True, shadow=True)

    if id == 0:
        ax23 = plt.subplot(3, 2, 3 + id)
    else:
        ax24 = plt.subplot(3, 2, 3 + id, sharey=ax23)
    plt.plot([x for x in range(1, 11)], avgspec[::2], 's-', lw=2, label="Specificity TP FN ")
    plt.plot([x for x in range(1, 11)], avgspec[1::2], 'p-', lw=2, label="Specificity AVG")
    if id == 0:
        plt.ylabel('Specificity')

    plt.xlim(0, 11)
    plt.ylim(0.95, 1.02)

    if id == 0:
        box = ax23.get_position()
        ax23.set_position([box.x0, box.y0 + box.height * 0.2,
                           box.width, box.height * 0.8])

        # Put a legend below current ax2is
        ax23.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                    fancybox=True, shadow=True)

    else:
        box = ax24.get_position()
        ax24.set_position([box.x0, box.y0 + box.height * 0.2,
                           box.width, box.height * 0.8])

        # Put a legend below current ax2is
        ax24.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                    fancybox=True, shadow=True)

    if id == 0:
        ax25 = plt.subplot(3, 2, 5 + id)
    else:
        ax26 = plt.subplot(3, 2, 5 + id, sharey=ax25)
    plt.plot([x for x in range(1, 11)], avgprec[::2], 's-', lw=2, label="Precision TP FN ")
    plt.plot([x for x in range(1, 11)], avgprec[1::2], 'p-', lw=2, label="Precision AVG")
    if id == 0:
        plt.ylabel('Precision')
    plt.xlabel('Procent klasy mniejszosciowej')
    plt.xlim(0, 11)

    if id == 0:
        box = ax25.get_position()
        ax25.set_position([box.x0, box.y0 + box.height * 0.1,
                           box.width, box.height * 0.9])

        # Put a legend below current ax2is
        ax25.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
                    fancybox=True, shadow=True)

    else:
        box = ax26.get_position()
        ax26.set_position([box.x0, box.y0 + box.height * 0.1,
                           box.width, box.height * 0.9])

        # Put a legend below current ax2is
        ax26.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),
                    fancybox=True, shadow=True)
plt.figure(1)

plt.savefig(os.path.join(path, 'wyniki/wykresy_zdjecia/stdwsk.png'))
plt.figure(2)
plt.savefig(os.path.join(path, 'wyniki/wykresy_zdjecia/wsk.png'))
plt.show()
raw_input()
