#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import Rnn as Rn
import NeuralNetwork as Nn


def main(verbose=1):
    fp = h5.File(os.path.join('.', 'testdata02.h5'), 'r')
    x = fp['/train/data'].value
    y = fp['/train/label'].value
    num_test = int(fp['/test/num'].value)
    test_x = []
    for i in range(num_test):
        test_x.append(fp['/test/data{}'.format(i)].value)
    fp.close()

    hid_lr = 0.4
    hid_mt = 0.5
    hid_wd = 0.0001
    out_lr = 0.05
    out_mt = 0.5
    out_wd = 0.0001
    ep = 50
    window = 0
    stride = 0

    hid_type = Nn.LAYER_TYPE_TANH
    hid_size = y.shape[1]
    hid_param = {
        'output_size': hid_size,
        'learning_rate': hid_lr, 'momentum': hid_mt, 'weight_decay': hid_wd
    }
    out_type = Nn.LAYER_TYPE_IDENTITY
    out_param = {
    #    'output_size': y.shape[1],
    #    'learning_rate': out_lr, 'momentum': out_mt, 'weight_decay': out_wd
    }
    rnn = Rn.Rnn(hid_type, hid_param, hid_size, out_type, out_param)
    rnn.train(x, y, ep, window, stride, verbose)

    idx = 2
    p = rnn.simulate(test_x[idx])
    t = np.arange(test_x[idx].shape[0])
    num_feature = p.shape[1] - 1
    plt.subplot(2, 1, 1)
    plt.plot(t, test_x[idx])
    plt.subplot(2, 1, 2)
    for i in range(num_feature):
        plt.plot(t, p[:, i], label='{}'.format(i))
    plt.legend(prop={'size': 7})
    plt.ylim(0, 1.1)
    plt.show()


if __name__ == "__main__":
    main()

