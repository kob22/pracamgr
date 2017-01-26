from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from texttable import Texttable
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
import os
from sklearn import preprocessing
"""
Base IO code for all datasets
"""

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2010 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

class Bunch(dict):

    def __init__(self, data, target):
        dict.__init__(self, data=data, target=target)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass

    def shuffle2(self, random_state=0):
        self['data'], self['target'] = shuffle(self['data'], self['target'], random_state=random_state)


path = os.path.dirname(os.path.abspath(__file__))

def importfile(file):
    np.set_printoptions(threshold=np.nan)

    datafile = np.genfromtxt(file, delimiter=",", dtype="i8,i8,S1")

    data = datafile[:,:-1]
    target = datafile[:,-1]

    return Bunch(data=data, target = target)


def load_german():
    datafile = np.loadtxt(os.path.join(path, "files/german.data"), dtype='uint8')

    data = datafile[:, :-1]
    target = datafile[:, -1]
    lb = preprocessing.LabelEncoder()
    lb.fit(target)
    target1 = lb.transform(target)

    return Bunch(data=data, target=target1)


def load_haberman():
    datafile = np.loadtxt(os.path.join(path, "files/haberman.data"), delimiter=",", dtype='uint8')

    data = datafile[:, :-1]
    target = datafile[:, -1]
    lb = preprocessing.LabelEncoder()
    lb.fit(target)
    target1 = lb.transform(target)

    return Bunch(data=data, target=target1)


def load_transfusion():
    datafile = np.loadtxt(os.path.join(path, "files/transfusion.data"), delimiter=",", dtype='uint16')
    data = datafile[:, :-1]
    target = datafile[:, -1]

    lb = preprocessing.LabelEncoder()
    lb.fit(target)
    target1 = lb.transform(target)
    return Bunch(data=data, target=target1)


def load_ionosphere():
    data = np.loadtxt(os.path.join(path, "files/ionosphere.data"), delimiter=",", usecols=[x for x in range(34)],
                      dtype='float')
    filetarget = np.loadtxt(os.path.join(path, "files/ionosphere.data"), delimiter=",", usecols=[34], dtype='|S1')
    target = []
    for line in filetarget:
        if line == 'b':
            target.append(0)
        elif line == 'g':
            target.append(1)
    return Bunch(data=data, target=np.asarray(target))


def load_balance_scale():
    data = np.loadtxt(os.path.join(path, "files/balance-scale.data"), delimiter=",", usecols=[x for x in range(1, 5)],
                      dtype='uint16')
    filetarget = np.loadtxt(os.path.join(path, "files/balance-scale.data"), delimiter=",", usecols=[0], dtype='|S1')
    target = []
    for line in filetarget:
        if line == 'R' or line == 'L':
            target.append(0)
        elif line == 'B':
            target.append(1)
    return Bunch(data=data, target=np.asarray(target))


def load_bupa():
    datafile = np.loadtxt(os.path.join(path, "files/bupa.data"), dtype='float')

    data = datafile[:, :-1]
    target = datafile[:, -1]
    lb = preprocessing.LabelEncoder()
    lb.fit(target)
    target1 = lb.transform(target)
    return Bunch(data=data, target=target1)


def load_car():
    datafile = np.loadtxt(os.path.join(path, "files/car.data"), delimiter=",", dtype='|S5')
    mapa = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1, '5more': 5, 'more': 5, 'small': 1, 'big': 3,
            '2': 2, '3': 3, '4': 4, 'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 1}
    data = []
    target = []
    for line in datafile:
        temp = ([mapa[letter] for letter in line])
        data.append(temp[:-1])
        target.append(temp[-1])
    return Bunch(data=np.asarray(data), target=np.asarray(target))


def load_cmc():
    datafile = np.loadtxt(os.path.join(path, "files/cmc.data"), delimiter=",", dtype='uint8')

    data = datafile[:, :-1]
    targettemp = datafile[:, -1]
    target = []
    for line in targettemp:
        if line == 1 or line == 3:
            target.append(0)
        elif line == 2:
            target.append(1)

    return Bunch(data=data, target=np.asarray(target))


def load_ecoli():
    data = np.loadtxt(os.path.join(path, "files/ecoli.data"), usecols=[x for x in range(1, 8)], dtype='float')
    filetarget = np.loadtxt(os.path.join(path, "files/ecoli.data"), usecols=[8], dtype='|S3')
    target = []
    classes = ['cp', 'im', 'pp', 'om', 'omL', 'imL', 'imS']
    for line in filetarget:
        if any(line in s for s in classes):
            target.append(0)
        elif line == 'imU':
            target.append(1)
    return Bunch(data=data, target=np.asarray(target))


def load_glass():
    data = np.loadtxt(os.path.join(path, "files/glass.data"), delimiter=",", usecols=[x for x in range(1, 10)],
                      dtype='float')
    filetarget = np.loadtxt(os.path.join(path, "files/glass.data"), delimiter=",", usecols=[10], dtype='uint8')
    target = []
    for line in filetarget:
        if line in (1, 2) or line in range(4, 8):
            target.append(0)
        elif line == 3:
            target.append(1)
    return Bunch(data=data, target=np.asarray(target))


def load_new_thyroid():
    datafile = np.loadtxt(os.path.join(path, "files/new-thyroid.data"), delimiter=",", dtype='float')
    data = datafile[:, 1:]
    targettemp = datafile[:, 0]
    target = []
    for item in targettemp:
        if item in (1, 2):
            target.append(0)
        elif item == 3:
            target.append(1)

    return Bunch(data=data, target=np.asarray(target, dtype='uint8'))


def load_seeds():
    datafile = np.loadtxt(os.path.join(path, "files/seeds.data"), dtype='float')
    data = datafile[:, :-1]
    targettemp = datafile[:, -1]
    target = []
    for item in targettemp:
        if item in (1, 2):
            target.append(0)
        elif item == 3:
            target.append(1)

    return Bunch(data=data, target=np.asarray(target, dtype='uint8'))


def load_solar_flare():
    data1 = np.loadtxt(os.path.join(path, "files/solar-flare.data"), usecols=[1, 2], dtype='|S1')
    data2 = np.loadtxt(os.path.join(path, "files/solar-flare.data"), usecols=[x for x in range(3, 11)], dtype='uint8')
    targettemp = np.loadtxt(os.path.join(path, "files/solar-flare.data"), usecols=[0], dtype='|S1')
    target = []
    datatemp = []
    classes = ['A', 'B', 'C', 'D', 'E', 'H']
    classesatr = {'X': 1, 'R': 2, 'S': 3, 'A': 4, 'H': 5, 'K': 6, 'O': 1, 'I': 2, 'C': 3}
    for line in data1:
        datatemp.append([classesatr[letter] for letter in line])

    for line in targettemp:
        if any(line in s for s in classes):
            target.append(0)
        elif line == 'F':
            target.append(1)
    return Bunch(data=np.concatenate((datatemp, data2), axis=1), target=np.asarray(target))


def load_vehicle():
    data = np.loadtxt(os.path.join(path, "files/vehicle.data"), usecols=[x for x in range(18)], dtype='uint8')
    targettemp = np.loadtxt(os.path.join(path, "files/vehicle.data"), usecols=[18], dtype='|S4')
    target = []
    classes = ['opel', 'saab', 'bus']

    for line in targettemp:
        if any(line in s for s in classes):
            target.append(0)
        elif line == 'van':
            target.append(1)
    return Bunch(data=data, target=np.asarray(target))


def load_vertebal():
    data = np.loadtxt(os.path.join(path, "files/vertebal.data"), delimiter=',', usecols=[x for x in range(6)],
                      dtype='float')
    targettemp = np.loadtxt(os.path.join(path, "files/vertebal.data"), delimiter=',', usecols=[6], dtype='|S8')
    target = []

    for line in targettemp:
        if line == 'Abnormal':
            target.append(0)
        elif line == 'Normal':
            target.append(1)
    return Bunch(data=data, target=np.asarray(target))


def load_yeastME3():
    data = np.loadtxt(os.path.join(path, "files/yeast.data"), usecols=[x for x in range(1, 9)], dtype='float')
    targettemp = np.loadtxt(os.path.join(path, "files/yeast.data"), usecols=[9], dtype='|S3')
    target = []
    classes = ['CYT', 'ERL', 'EXC', 'ME1', 'ME2', 'MIT', 'NUC', 'POX', 'VAC']
    for line in targettemp:
        if any(line in s for s in classes):
            target.append(0)
        elif line == 'ME3':
            target.append(1)
    return Bunch(data=data, target=np.asarray(target))


def load_yeastME2():
    data = np.loadtxt(os.path.join(path, "files/yeast.data"), usecols=[x for x in range(1, 9)], dtype='float')
    targettemp = np.loadtxt(os.path.join(path, "files/yeast.data"), usecols=[9], dtype='|S3')
    target = []
    classes = ['CYT', 'ERL', 'EXC', 'ME1', 'ME3', 'MIT', 'NUC', 'POX', 'VAC']
    for line in targettemp:
        if any(line in s for s in classes):
            target.append(0)
        elif line == 'ME2':
            target.append(1)
    return Bunch(data=data, target=np.asarray(target))


def load_yeastME1():
    data = np.loadtxt(os.path.join(path, "files/yeast.data"), usecols=[x for x in range(1, 9)], dtype='float')
    targettemp = np.loadtxt(os.path.join(path, "files/yeast.data"), usecols=[9], dtype='|S3')
    target = []
    classes = ['CYT', 'ERL', 'EXC', 'ME2', 'ME3', 'MIT', 'NUC', 'POX', 'VAC']
    for line in targettemp:
        if any(line in s for s in classes):
            target.append(0)
        elif line == 'ME1':
            target.append(1)
    return Bunch(data=data, target=np.asarray(target))


def load_abalone0_4():
    data1 = np.loadtxt(os.path.join(path, "files/abalone.data"), delimiter=',', usecols=[0], dtype='|S1')
    data2 = np.loadtxt(os.path.join(path, "files/abalone.data"), delimiter=',', usecols=[x for x in range(1, 9)],
                       dtype='float')

    data1temp = []
    mapa = {'I': 0, 'M': 1, 'F': 2}
    for item in data1:
        data1temp.append([mapa[item]])

    data = np.concatenate((data1temp, data2[:, :-1]), axis=1)

    targettemp = data2[:, -1]
    target = []
    for item in targettemp:
        if item in range(1, 5):
            target.append(0)
        elif item in range(5, 30):
            target.append(1)
    return Bunch(data=data, target=np.asarray(target, dtype='uint8'))


def load_abalone16_29():
    data1 = np.loadtxt(os.path.join(path, "files/abalone.data"), delimiter=',', usecols=[0], dtype='|S1')
    data2 = np.loadtxt(os.path.join(path, "files/abalone.data"), delimiter=',', usecols=[x for x in range(1, 9)],
                       dtype='float')

    data1temp = []
    mapa = {'I': 0, 'M': 1, 'F': 2}
    for item in data1:
        data1temp.append([mapa[item]])

    data = np.concatenate((data1temp, data2[:, :-1]), axis=1)

    targettemp = data2[:, -1]
    target = []
    for item in targettemp:
        if item in range(16, 30):
            target.append(0)
        elif item in range(1, 16):
            target.append(1)
    return Bunch(data=data, target=np.asarray(target, dtype='uint8'))


def load_abalone0_4_16_29():
    data1 = np.loadtxt(os.path.join(path, "files/abalone.data"), delimiter=',', usecols=[0], dtype='|S1')
    data2 = np.loadtxt(os.path.join(path, "files/abalone.data"), delimiter=',', usecols=[x for x in range(1, 9)],
                       dtype='float')

    data1temp = []
    mapa = {'I': 0, 'M': 1, 'F': 2}
    for item in data1:
        data1temp.append([mapa[item]])

    data = np.concatenate((data1temp, data2[:, :-1]), axis=1)

    targettemp = data2[:, -1]
    target = []
    for item in targettemp:
        if item in range(16, 30) or item in range(1, 5):
            target.append(0)
        elif item in range(4, 16):
            target.append(1)

    return Bunch(data=data, target=np.asarray(target, dtype='uint8'))


# missing

def load_breast_cancer(imput_strategy='median'):
    datafile = np.genfromtxt(os.path.join(path, "files/missing/breast-cancer.data"), missing_values='?', delimiter=',',
                             dtype='|S20')

    maps = [
        {'10-19': 15, '20-29': 25, '30-39': 35, '40-49': 45, '50-59': 55, '60-69': 65, '70-79': 75, '80-89': 85,
         '90-99': 95},
        {'lt40': 1, 'ge40': 2, 'premeno': 3},
        {'0-4': 2, '5-9': 7, '10-14': 12, '15-19': 17, '20-24': 22, '25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42,
         '45-49': 47, '50-54': 52, '55-59': 57},
        {'0-2': 1, '3-5': 4, '6-8': 7, '9-11': 10, '12-14': 13, '15-17': 16, '18-20': 19, '21-23': 22, '24-26': 25,
         '27-29': 28, '30-32': 31, '33-35': 34, '36-39': 37},
        {'yes': 1, 'no': 0},
        {'1': 1, '2': 2, '3': 3},
        {'left': 0, 'right': 1},
        {'left_up': 1, 'left_low': 2, 'right_up': 3, 'right_low': 4, 'central': 5},
        {'yes': 1, 'no': 0},
        {'recurrence-events': 1, 'no-recurrence-events': 0}
    ]
    datatemp = []
    target = []
    for line in datafile:
        templine = []
        for item, dict in zip(line, maps):

            if item == '?':
                templine.append(np.NaN)
            else:
                templine.append(dict[item])
        datatemp.append(templine[:-1])
        target.append(templine[-1])

    imp = Imputer(missing_values='NaN', strategy=imput_strategy, axis=0)
    imp = imp.fit(datatemp)
    imputed_data = imp.transform(datatemp)
    return Bunch(data=imputed_data, target=np.asarray(target, dtype='uint8'))


def load_hepatitis(imput_strategy='median'):
    datafile = np.genfromtxt(os.path.join(path, "files/missing/hepatitis.data"), missing_values='?', delimiter=',',
                             dtype='float')

    data = datafile[:, 1:]
    target = datafile[:, 0]

    imp = Imputer(missing_values='NaN', strategy=imput_strategy, axis=0)
    imp = imp.fit(data)
    imputed_data = imp.transform(data)
    lb = preprocessing.LabelEncoder()
    lb.fit(target)
    target1 = lb.transform(target)
    return Bunch(data=imputed_data, target=np.asarray(target1, dtype='uint8'))


def load_hear_cleveland(imput_strategy='median'):
    datafile = np.genfromtxt(os.path.join(path, "files/missing/heart-cleveland.data"), missing_values='?',
                             delimiter=',', dtype='float')

    data = datafile[:, :-1]
    targettemp = datafile[:, -1]
    target = []
    for item in targettemp:
        if item in (0, 1, 2, 4):
            target.append(0)
        elif item == 3:
            target.append(1)

    imp = Imputer(missing_values='NaN', strategy=imput_strategy, axis=0)
    imp = imp.fit(data)
    imputed_data = imp.transform(data)
    return Bunch(data=imputed_data, target=np.asarray(target, dtype='uint8'))


def load_postoperative(imput_strategy='median'):
    datafile = np.genfromtxt(os.path.join(path, "files/missing/postoperative.data"), missing_values='?', delimiter=',',
                             dtype='|S4,|S4,|S10,|S4,|S10,|S10,|S10,i4,|S2')

    maps = [

        {'high': 37.5, 'mid': 36.5, 'low': 35.5},
        {'high': 37, 'mid': 36, 'low': 34.5},
        {'excellent': 98, 'good': 94, 'fair': 85, 'poor': 79},
        {'high': 3, 'mid': 2, 'low': 1},
        {'stable': 3, 'mod-stable': 2, 'unstable': 1},
        {'stable': 3, 'mod-stable': 2, 'unstable': 1},
        {'stable': 3, 'mod-stable': 2, 'unstable': 1}
    ]
    data = []
    target = []
    for line in datafile:

        datatemp = []
        temp = []
        for item in line:
            temp.append(item)

        for item, dict in zip(temp[:-2], maps):
            datatemp.append(dict[item])
        if temp[-2] == -1:
            datatemp.append(np.NaN)
        else:
            datatemp.append(line[-2])
        if temp[-1] in ['A', 'I']:
            target.append(0)
        else:
            target.append(1)
        data.append(datatemp)
        imp = Imputer(missing_values='NaN', strategy=imput_strategy, axis=0)
        imp = imp.fit(data)
        imputed_data = imp.transform(data)
    return Bunch(data=imputed_data, target=np.asarray(target, dtype='uint8'))


def load_horse_colic(imput_strategy='median'):
    cols = [x for x in range(24)]
    cols.remove(2)

    datafile = np.genfromtxt(os.path.join(path, "files/missing/horse-colic.data"), missing_values='?', usecols=cols)

    data = datafile[:, :-1]
    target = datafile[:, -1]

    imp = Imputer(missing_values='NaN', strategy=imput_strategy, axis=0)
    imp = imp.fit(data)
    imputed_data = imp.transform(data)
    lb = preprocessing.LabelEncoder()
    lb.fit(target)
    target1 = lb.transform(target)
    return Bunch(data=imputed_data, target=np.asarray(target1, dtype='uint8'))

def print_info(target):
    total_n_el = target.size
    print("Liczba elementow: %s" % total_n_el)
    groups, counts = np.unique(target, return_counts=True)
    percent_total = []
    for quantity in counts:
        percent_total.append(quantity / total_n_el)

    rows = [(group, quantity, percent) for group, quantity, percent in zip(groups, counts, percent_total)]
    cols_name = ['Klasa', 'Liczba wystapien', 'Procent calosci']
    table = Texttable()
    table.add_rows([cols_name, rows[0], rows[1]])
    print(table.draw())
