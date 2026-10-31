# -*- coding: utf-8 -*-
# Origin: VicidominiLab/BrightEyes-FFS, src/brighteyes_ffs/tools/savevar.py at f6d640d.
"""
Created on Fri Apr 12 12:01:51 2019

@author: SPAD-fcs
"""

import pickle
from .checkfname import checkfname


def savevar(var, fname):
    fname = checkfname(fname, 'pickle')
    with open(fname, 'wb') as f:
        pickle.dump(var, f)


def openvar(fname):
    fname = checkfname(fname, 'pickle')
    with open(fname, 'rb') as f:
        data = pickle.load(f)
        return data
