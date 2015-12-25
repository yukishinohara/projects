#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import h5py as h5
import Dbn as dl
import OutputLayer as ol


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

    # Params
    hidden = np.array([50, 50, 50])
    ita = 0.15
    epoch = 100
    momentum = 0.5
    weight_decay = 0.001
    batch_size = 30

    # Test it
    dbn = dl.Dbn(instances=x_train, targets=y_train,
                 hidden_sizes=hidden,
                 lr=ita, mt=momentum, wd=weight_decay, bs=batch_size
                 )

    if verbose >= 1:
        print('Pre Train:')
    h_out = dbn.pre_train(epoch=epoch, verbose=verbose)
    if verbose >= 1:
        print(h_out)
        print('Fine Tune:')
    dbn.fine_tune(h_out, epoch=epoch, verbose=verbose)

    p = dbn.sim(x_test)
    p_fmt = np.argmax(p, axis=1)
    t_fmt = np.argmax(y_test, axis=1)

    err_num = np.count_nonzero(t_fmt - p_fmt)
    total_num = p_fmt.size

    print('')
    print(p_fmt)
    print(t_fmt)
    print('Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))

    ann = ol.OutputLayer(x_test, y_test, lr=ita, mt=momentum, wd=weight_decay, bs=batch_size)
    for ep in range(epoch):
        ann.train()
    p = ann.sim(x_test)
    p_fmt = np.argmax(p, axis=1)
    err_num = np.count_nonzero(t_fmt - p_fmt)
    total_num = p_fmt.size
    print('')
    print(p_fmt)
    print(t_fmt)
    print('Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))


if __name__ == "__main__":
    main()

