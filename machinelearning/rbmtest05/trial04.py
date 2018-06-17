#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import h5py as h5
import RbmVectorized as rl
import Rbm as rr
import crossval as cv
import time


def train_with(x, y, params, verbose):
    # Parameters
    ita = None
    epoch = None
    hidden = None
    momentum = None
    weight_decay = None
    batch_size = None
    if params is not None:
        ita = params.get('ita')
        epoch = params.get('epoch')
        hidden = params.get('hidden')
        momentum = params.get('momentum')
        weight_decay = params.get('weight_decay')
        batch_size = params.get('batch_size')
    ita = 0.8 if ita is None else ita
    epoch = 100 if epoch is None else epoch
    hidden = 10 if hidden is None else hidden
    momentum = 0. if momentum is None else momentum
    weight_decay = 0. if weight_decay is None else weight_decay
    batch_size = 0 if batch_size is None else batch_size

    # Organize instances
    y_x = np.c_[y, x]
    m = y_x[0, :].size

    # Generate and train RBM
    rbm = rr.Rbm(D=y_x, m=m, n=hidden, lr=ita, mt=momentum, wd=weight_decay, bs=batch_size)
    for ep in range(epoch):
        rbm.train()
        if verbose >= 5:
            print(' ep={}, E={}'.format(ep, rbm.log_pl_all()))
        elif verbose >= 3 and ep % 10 == 1:
            print('\r ep={}, E={}'.format(ep, rbm.cost()))
        elif verbose >= 2:
            print('\r ep={}'.format(ep), end="")
    return rbm


def train_with_vector(x, y, params, verbose):
    # Parameters
    ita = None
    epoch = None
    hidden = None
    momentum = None
    weight_decay = None
    batch_size = None
    if params is not None:
        ita = params.get('ita')
        epoch = params.get('epoch')
        hidden = params.get('hidden')
        momentum = params.get('momentum')
        weight_decay = params.get('weight_decay')
        batch_size = params.get('batch_size')
    ita = 0.8 if ita is None else ita
    epoch = 100 if epoch is None else epoch
    hidden = 10 if hidden is None else hidden
    momentum = 0. if momentum is None else momentum
    weight_decay = 0. if weight_decay is None else weight_decay
    batch_size = 0 if batch_size is None else batch_size

    # Organize instances
    y_x = np.c_[y, x]
    m = y_x[0, :].size

    # Generate and train RBM
    rbm = rl.RbmVectorized(D=y_x, m=m, n=hidden, lr=ita, mt=momentum, wd=weight_decay, bs=batch_size)
    for ep in range(epoch):
        rbm.train()
        if verbose >= 5:
            print(' ep={}, E={}'.format(ep, rbm.log_pl_all()))
        elif verbose >= 3 and ep % 10 == 1:
            print('\r ep={}, E={}'.format(ep, rbm.cost()))
        elif verbose >= 2:
            print('\r ep={}'.format(ep), end="")
    return rbm


def test_with(rbm, x, y, params, verbose):
    # Organize test data
    feat_size = y[0, :].size
    test_size = y[:, 0].size
    z = np.zeros((test_size, feat_size))
    z_x = np.c_[z, x]

    # Test
    p = (rbm.sim2(z_x))[:, 0:feat_size]
    if verbose >= 4:
        for l in range(test_size):
            energy = rbm.log_pl_precise(z_x[l:l+1, :])
            print('x={}, F={}'.format(x[l, :], energy))

    return p


def main(verbose=2):
    # Read test data 01
    fp = h5.File('testdata01.h5', 'r')
    x = fp['/clean/x'].value
    y = fp['/clean/y'].value
    fp.close()

    # Params
    params = {
        'hidden': 11,
        'ita': 0.1,
        'epoch': 100,
        'momentum': 0.7,
        'weight_decay': 0.001,
        'batch_size': 129
    }

    # Test it
    start = time.time()
    [er1, _] = cv.cross_validation(x, y, train_with, test_with, k=1, params=params, verbose=verbose)
    rrtime = time.time() - start

    start = time.time()
    [er2, _] = cv.cross_validation(x, y, train_with_vector, test_with, k=1, params=params, verbose=verbose)
    rltime = time.time() - start

    print('Normal: time={}, er={}%'.format(rrtime, er1*100))
    print('Vector: time={}, er={}%'.format(rltime, er2*100))

if __name__ == "__main__":
    main()

