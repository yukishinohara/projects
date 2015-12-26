#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import Dbn as dl
import OutputLayer as ol
import mnistloader as mn
import shelve
import zipfile
import os


def main(verbose=2):
    loader = mn.MNISTloader('__data__')
    [x, y] = loader.load_train_data(msize=5000)
    print(x.shape)
    print(y.shape)

    # Params
    hidden = np.array([280, 280, 280])
    ita = 0.1
    epoch = 300
    momentum = 0.7
    weight_decay = 0.001
    batch_size = 50

    # Train
    dbn = dl.Dbn(
        instances=x, targets=y,
        hidden_sizes=hidden,
        lr=ita, mt=momentum, wd=weight_decay, bs=batch_size
    )
    h_out = dbn.pre_train(epoch=epoch, verbose=verbose)
    dbn.fine_tune(h_out, epoch=epoch, verbose=verbose)
    ann = ol.OutputLayer(
        x, y,
        lr=ita, mt=momentum, wd=weight_decay, bs=batch_size
    )
    for ep in range(epoch):
        ann.train()

    # Test
    [xt, yt] = loader.load_test_data(msize=1000)
    p_dbn = dbn.sim(xt)
    p_ann = ann.sim(xt)
    y_fmt = np.argmax(yt, axis=1)
    p_dbn_fmt = np.argmax(p_dbn, axis=1)
    p_ann_fmt = np.argmax(p_ann, axis=1)

    print('')
    err = np.count_nonzero(y_fmt - p_dbn_fmt)
    err_rate = float(err) / y_fmt.size
    print(y_fmt)
    print(p_dbn_fmt)
    print('DBN: er={}% ({}/{})'.format(err_rate*100, err, y_fmt.size))
    err = np.count_nonzero(y_fmt - p_ann_fmt)
    err_rate = float(err) / y_fmt.size
    print(y_fmt)
    print(p_ann_fmt)
    print('ANN: er={}% ({}/{})'.format(err_rate*100, err, y_fmt.size))

    # Save
    dfile = shelve.open(os.path.join('.', '__trained__', 'dbn_mnist'))
    dfile['dbn'] = dbn
    dfile.close()
    zipFile = zipfile.ZipFile('dbn_mnist.zip', 'w', zipfile.ZIP_DEFLATED)
    zipFile.write(os.path.join('.', '__trained__', 'dbn_mnist.bak'))
    zipFile.write(os.path.join('.', '__trained__', 'dbn_mnist.dat'))
    zipFile.write(os.path.join('.', '__trained__', 'dbn_mnist.dir'))
    zipFile.close()


def main2(verbose=2):
    loader = mn.MNISTloader('..\\rbmtest05\\__data__')
    [xt, yt] = loader.load_test_data(msize=1000)

    # Load DBN
    zipFile = zipfile.ZipFile('dbn_mnist.zip', 'r')
    zipFile.extractall('.')
    zipFile.close()
    dfile = shelve.open(os.path.join('.', '__trained__', 'dbn_mnist'))
    dbn = dfile['dbn']
    dfile.close()
    p_dbn = dbn.sim(xt)
    y_fmt = np.argmax(yt, axis=1)
    p_dbn_fmt = np.argmax(p_dbn, axis=1)
    print('')
    err = np.count_nonzero(y_fmt - p_dbn_fmt)
    err_rate = float(err) / y_fmt.size
    print(y_fmt)
    print(p_dbn_fmt)
    print('DBN: er={}% ({}/{})'.format(err_rate*100, err, y_fmt.size))


if __name__ == "__main__":
    main2()

