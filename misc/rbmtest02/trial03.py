#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import h5py as h5
import Rbm
import crossval as cv


def train_with(x, y, params, verbose):
    # Parameters
    ita = None
    epoch = None
    hidden = None
    momentum = None
    weight_decay = None
    if params is not None:
        ita = params.get('ita')
        epoch = params.get('epoch')
        hidden = params.get('hidden')
        momentum = params.get('momentum')
        weight_decay = params.get('weight_decay')
    ita = 0.8 if ita is None else ita
    epoch = 100 if epoch is None else epoch
    hidden = 10 if hidden is None else hidden
    momentum = 0. if momentum is None else momentum
    weight_decay = 0. if weight_decay is None else weight_decay

    # Organize instances
    y_x = np.c_[y, x]
    m = y_x[0, :].size

    # Generate and train RBM
    rbm = Rbm.Rbm(D=y_x, m=m, n=hidden, a=ita, u=momentum, q=weight_decay)
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


def main(verbose=3):
    # Read test data 01
    fp = h5.File('testdata01.h5', 'r')
    x = fp['/clean/x'].value
    y = fp['/clean/y'].value
    fp.close()

    # Params
    params = {
        'hidden': 3,
        'ita': 0.75,
        'epoch': 100,
        'momentum': 0.6,
        'weight_decay': 0.001
    }

    # Test it
    [er, ans] = cv.cross_validation(x, y, train_with, test_with, k=10, params=params, verbose=verbose)
    if verbose >= 2:
        print('Overall Result: {}'.format(ans))
    print('Overall Err rate: {}%'.format(er*100))

if __name__ == "__main__":
    main()

