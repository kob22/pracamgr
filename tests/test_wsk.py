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
max_iter = 100
clf = tree.DecisionTreeClassifier()
folds = [StratifiedKFold(n_splits=10, random_state=5), KFold(n_splits=10, random_state=5)]
name_folds = ['Stratified K-fold, k=10', 'Unstratified K-fold, k=10']

fig1 = plt.figure(facecolor='white', figsize=(12, 14))
fig2 = plt.figure(facecolor='white', figsize=(12, 14))
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

            print(sensivity_calc(predictions, targets))

            sens.extend(sensivity_calc(predictions, targets))
            spec.extend(specif_calc(predictions, targets))
            prec.extend(precion_calc(predictions, targets))

        print(avgsens)

        stdsens.extend([np.std(sens[::2]), np.std(sens[1::2])])
        stdspec.extend([np.std(spec[::2]), np.std(spec[1::2])])
        stdprec.extend([np.std(prec[::2]), np.std(prec[1::2])])

        avgsens.extend([np.mean(sens[::2]), np.mean(sens[1::2])])
        avgspec.extend([np.mean(spec[::2]), np.mean(spec[1::2])])
        avgprec.extend([np.mean(prec[::2]), np.mean(prec[1::2])])
        print('----')
        print(sens)
        print(sens[::2])
        print(sens[1::2])
        print([np.std(sens[::2]), np.std(sens[1::2])])
        print([np.mean(sens[::2]), np.mean(sens[1::2])])
        print(avgsens)
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

    plt.figure(1)
    ax = plt.subplot(3, 2, 1 + id)

    plt.plot([x for x in range(1, 11)], stdsens[::2], 's-', lw=2, label="Sensitivity TP FN ")
    plt.plot([x for x in range(1, 11)], stdsens[1::2], 'p-', lw=2, label="Sensitivity AVG")
    if id == 0:
        plt.ylabel('Odchylenie standardowe')

    plt.xlim(0, 11)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True)

    ax = plt.subplot(3, 2, 3 + id)
    plt.plot([x for x in range(1, 11)], stdspec[::2], 's-', lw=2, label="Specificity TP FN ")
    plt.plot([x for x in range(1, 11)], stdspec[1::2], 'p-', lw=2, label="Specificity AVG")
    if id == 0:
        plt.ylabel('Odchylenie standardowe')

    plt.xlim(0, 11)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True)

    ax = plt.subplot(3, 2, 5 + id)
    plt.plot([x for x in range(1, 11)], stdprec[::2], 's-', lw=2, label="Precision TP FN ")
    plt.plot([x for x in range(1, 11)], stdprec[1::2], 'p-', lw=2, label="Precision AVG")
    if id == 0:
        plt.ylabel('Odchylenie standardowe')
    plt.xlabel('Procent klasy mniejszosciowej')
    plt.xlim(0, 11)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=True)

    plt.figure(2)
    ax = plt.subplot(3, 2, 1 + id)

    plt.plot([x for x in range(1, 11)], avgsens[::2], 's-', lw=2, label="Sensitivity TP FN ")
    plt.plot([x for x in range(1, 11)], avgsens[1::2], 'p-', lw=2, label="Sensitivity AVG")
    if id == 0:
        plt.ylabel('Sensitivity')

    plt.xlim(0, 11)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True)

    ax = plt.subplot(3, 2, 3 + id)
    plt.plot([x for x in range(1, 11)], avgspec[::2], 's-', lw=2, label="Specificity TP FN ")
    plt.plot([x for x in range(1, 11)], avgspec[1::2], 'p-', lw=2, label="Specificity AVG")
    if id == 0:
        plt.ylabel('Specificity')

    plt.xlim(0, 11)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True)

    ax = plt.subplot(3, 2, 5 + id)
    plt.plot([x for x in range(1, 11)], avgprec[::2], 's-', lw=2, label="Precision TP FN ")
    plt.plot([x for x in range(1, 11)], avgprec[1::2], 'p-', lw=2, label="Precision AVG")
    if id == 0:
        plt.ylabel('Precision')
    plt.xlabel('Procent klasy mniejszosciowej')
    plt.xlim(0, 11)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=True)
plt.figure(1)
plt.savefig(os.path.join(path, 'wyniki/wykresy_zdjecia/stdwsk.png'))
plt.figure(2)
plt.savefig(os.path.join(path, 'wyniki/wykresy_zdjecia/wsk.png'))
plt.show()
raw_input()
