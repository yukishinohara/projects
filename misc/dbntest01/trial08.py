#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import h5py as h5
import Dbn as dl


def main(verbose=1):
    # Read test data 01
    fp = h5.File('testdata01.h5', 'r')
    x = fp['/clean/x'].value
    y = fp['/clean/y'].value
    fp.close()

    train_size = 900
    x_train = x[:train_size, :]
    y_train = y[:train_size, :]
    x_test = x[train_size:, :]
    y_test = y[train_size:, :]

    dbn = dl.Dbn()
    # For Pre-training
    learning_rate = 0.15
    epoch = 100
    momentum = 0.5
    weight_decay = 0.003
    batch_size = 50
    dbn.pre_train(x_train, y_train, hidden_sizes=[45, 45],
                  learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                  batch_size=batch_size, epoch=epoch, verbose=verbose)

    # For Fine-tuning
    learning_rate = 0.07
    epoch = 250
    momentum = 0.5
    weight_decay = 0.001
    batch_size = 50
    dbn.fine_tune(x_train, y_train,
                  learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                  batch_size=batch_size, epoch=epoch, verbose=verbose)

    p = dbn.simulate(x_test)
    p_fmt = np.argmax(p, axis=1)
    t_fmt = np.argmax(y_test, axis=1)
    err_num = np.count_nonzero(t_fmt - p_fmt)
    total_num = p_fmt.size

    print('')
    print(p_fmt)
    print(t_fmt)
    print('Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))


if __name__ == "__main__":
    main()

