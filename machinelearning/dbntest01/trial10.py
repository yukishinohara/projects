#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import Dbn as dl
import mnistloader as mn
import shelve
import zipfile
import os


def main(verbose=2):
    loader = mn.MNISTloader('__data__')
    [x, y] = loader.load_train_data(msize=10000)
    print(x.shape)
    print(y.shape)

    dbn = dl.Dbn()
    # For Pre-training
    learning_rate = 0.1
    epoch = 240
    momentum = 0.6
    weight_decay = 0.001
    batch_size = 40
    dbn.pre_train(x, y, hidden_sizes=[300, 300],
                  learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                  batch_size=batch_size, epoch=epoch, verbose=verbose)

    # For Fine-tuning
    learning_rate = 0.05
    epoch = 160
    momentum = 0.5
    weight_decay = 0.001
    batch_size = 40
    dbn.fine_tune(x, y,
                  learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                  batch_size=batch_size, epoch=epoch, verbose=verbose)

    # Test
    [xt, yt] = loader.load_test_data(msize=1000)
    p_dbn = dbn.simulate(xt)
    y_fmt = np.argmax(yt, axis=1)
    p_dbn_fmt = np.argmax(p_dbn, axis=1)

    print('')
    err = np.count_nonzero(y_fmt - p_dbn_fmt)
    err_rate = float(err) / y_fmt.size
    print(y_fmt)
    print(p_dbn_fmt)
    print('Result: er={}% ({}/{})'.format(err_rate*100, err, y_fmt.size))

    # Save
    dfile = shelve.open(os.path.join('.', '__trained__', 'dbn_mnist'))
    dfile['dbn'] = dbn
    dfile.close()
    zipFile = zipfile.ZipFile('dbn_mnist.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(os.path.join('.', '__trained__')):
        for file in files:
            zipFile.write(os.path.join(root, file))
    zipFile.close()


def main2(verbose=2):
    loader = mn.MNISTloader(os.path.join('.', '__data__'))
    [xt, yt] = loader.load_test_data(msize=1000)

    # Load DBN
    zipFile = zipfile.ZipFile('dbn_mnist.zip', 'r')
    zipFile.extractall('.')
    zipFile.close()
    dfile = shelve.open(os.path.join('.', '__trained__', 'dbn_mnist'))
    dbn = dfile['dbn']
    dfile.close()
    p_dbn = dbn.simulate(xt)
    y_fmt = np.argmax(yt, axis=1)
    p_dbn_fmt = np.argmax(p_dbn, axis=1)
    print('')
    err = np.count_nonzero(y_fmt - p_dbn_fmt)
    err_rate = float(err) / y_fmt.size
    print(y_fmt)
    print(p_dbn_fmt)
    print('Result: er={}% ({}/{})'.format(err_rate*100, err, y_fmt.size))


if __name__ == "__main__":
    main2()

