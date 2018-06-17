#!/usr/bin/env python

from __future__ import print_function
import h5py as h5
import Dbn as dl
import crossval as cv


def main(verbose=3):
    # Read test data 01
    fp = h5.File('testdata01.h5', 'r')
    x = fp['/noisy/x'].value
    y = fp['/noisy/y'].value
    fp.close()

    def train_with(x_train, y_train, parms, v=1):
        dbn = dl.Dbn()
        # For Pre-training
        learning_rate = 0.15
        epoch = 200
        momentum = 0.8
        weight_decay = 0.0005
        batch_size = 35
        dbn.pre_train(x_train, y_train, hidden_sizes=[45, 45],
                      learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                      batch_size=batch_size, epoch=epoch, verbose=v)
        # For Fine-tuning
        learning_rate = 0.07
        epoch = 250
        momentum = 0.5
        weight_decay = 0.001
        batch_size = 35
        dbn.fine_tune(x_train, y_train,
                      learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                      batch_size=batch_size, epoch=epoch, verbose=v)
        return dbn

    def test_with(dbn, x_test, y_test, parms, v):
        return dbn.simulate(x_test)

    [err, _] = cv.cross_validation(x, y, train_with, test_with, k=10, verbose=verbose)
    print('Overall Result: {}%'.format(err*100))


if __name__ == "__main__":
    main()

