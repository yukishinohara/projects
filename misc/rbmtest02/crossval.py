#!/usr/bin/env python

from __future__ import print_function
import numpy as np


# k-fold cross validation
# == evaluate with 1-fold test data set / train without any validation data set
def cross_validation(x, y, train_func, test_func, k=10, repeat=1, params=None, verbose=0):
    er = 0
    data_size = x[:, 0].size
    vint = np.vectorize(lambda l: k-1 if l >= k else int(l))
    indices = np.arange(0, data_size)
    fold_size = int(data_size / k)
    indices = vint(indices / fold_size)
    ers = np.zeros((repeat, k))

    if k < 2:
        indices = np.arange(0, data_size)
        fold_size = int(data_size / 10)
        indices = 0 + (indices > fold_size)

    for q in range(repeat):
        for s in range(k):
            x_tran = x[indices != s]
            y_tran = y[indices != s]
            x_test = x[indices == s]
            y_test = y[indices == s]
            if verbose >= 1:
                print('Loop: {}/{}'.format((q*k + s), k*repeat))
            if verbose >= 2:
                print(' Train size: {}'.format(x_tran[:, 0].size))
                print(' Test size : {}'.format(x_test[:, 0].size))

            model = train_func(x_tran, y_tran, params, verbose)
            pred = test_func(model, x_test, y_test, params, verbose)

            t = np.argmax(y_test, axis=1)
            p = np.argmax(pred, axis=1)
            ers[q, s] = np.count_nonzero(t - p) / t.size
            er += ers[q, s]
            if verbose >= 3:
                print ('\n predict: {}'.format(p))
                print (' target : {}'.format(t))
            if verbose >= 1:
                print(' Err rate: {}%'.format(100 * ers[q, s]))

    er /= float(k*repeat)

    return [er,
            {
                'params': params,
                'err_rates': ers,
                'indices': indices
            }]

