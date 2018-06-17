#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import h5py as h5
import Rbm


def main(ita=0.7, epoch=60, verbose=2, hidden=4, test_prop=5):
    # Read test data 01
    fp = h5.File('testdata01.h5', 'r')
    x = fp['/clean/x'].value
    y = fp['/clean/y'].value
    data_size = x[:, 0].size
    attr_size = x[0, :].size
    feat_size = y[0, :].size
    fp.close()

    # Organize data
    tran_size = int((data_size * (100 - test_prop)) / 100)
    test_size = data_size - tran_size
    z = np.zeros((test_size, feat_size))
    inptd = np.c_[y[0:tran_size, :], x[0:tran_size, :]]
    testd = np.c_[z, x[tran_size:, :]]  # error if my calculation has mistake
    targy = y[tran_size:, :]

    # Generate RBM
    rbm = Rbm.Rbm(inptd, m=attr_size+feat_size, n=hidden, lr=ita)

    # Training
    for ep in range(epoch):
        rbm.train()
        if verbose >= 3:
            print ('ep={}, E={}'.format(ep, rbm.log_pl_all()))
        elif verbose >= 1:
            print ('\rep={}'.format(ep), end="")

    # Re-construct
    predy = (rbm.sim2(testd))[:, 0:feat_size]

    # Answer check phase
    t = np.argmax(targy, axis=1)
    p = np.argmax(predy, axis=1)
    print('')
    if verbose >= 2:
        for l in range(test_size):
            energy = rbm.log_pl_precise(testd[l:l+1, :])
            print ('p={}, y={}, F={}'.format(p[l], t[l], energy))
    if verbose >= 1:
        print ('predict = {}'.format(p))
        print ('target  = {}'.format(t))
    er = np.count_nonzero((t-p)) / test_size
    print ('ER = {}%'.format(er*100))

    return predy


if __name__ == "__main__":
    main()

