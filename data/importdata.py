import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle



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
    with open(file, 'r') as f:
        datafile = np.loadtxt(f)


    data = datafile[:,:-1]
    target = datafile[:,-1]

    return Bunch(data=data, target = target)