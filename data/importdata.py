from __future__ import division
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from texttable import Texttable


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

    def shuffle2(self):
        self['data'], self['target'] = shuffle(self['data'], self['target'], random_state=0)


def importfile(file):
    np.set_printoptions(threshold=np.nan)

    datafile = np.genfromtxt(file, delimiter=",", dtype="i8,i8,S1")

    data = datafile[:,:-1]
    target = datafile[:,-1]

    return Bunch(data=data, target = target)


def load_german():
    datafile = np.loadtxt("files/german.data", dtype='uint8')

    data = datafile[:, :-1]
    target = datafile[:, -1]
    return Bunch(data=data, target=target)


def load_haberman():
    datafile = np.loadtxt("files/haberman.data", delimiter=",", dtype='uint8')

    data = datafile[:, :-1]
    target = datafile[:, -1]

    return Bunch(data=data, target=target)


def load_transfusion():
    datafile = np.loadtxt("files/transfusion.data", delimiter=",", dtype='uint16')
    data = datafile[:, :-1]
    target = datafile[:, -1]

    return Bunch(data=data, target=target)


def load_ionosphere():
    datafile = np.loadtxt("files/ionosphere.data", delimiter=",", dtype='float')
    data = datafile[:, :-1]
    temp = datafile[:, -1]
    target = temp.astype("uint8")

    return Bunch(data=data, target=target)


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
