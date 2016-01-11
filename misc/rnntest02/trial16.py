#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import Rnn as Rn
import Cnn as Cn
import NeuralNetwork as Nn


def train_mlp(x, y, verbose):
    # Common hyper parameters
    sg_lr = 0.3
    sg_mt = 0.5
    sg_wd = 0.0001
    bs = 50
    ep = 100

    # Create model
    types = []
    params = []
    types.append(Nn.LAYER_TYPE_SIGMOID)
    params.append({
        'output_size': y.shape[1],
        'learning_rate': sg_lr, 'momentum': sg_mt, 'weight_decay': sg_wd
    })
    cnn = Cn.Cnn(types, params)

    # Training
    cnn.train(x, y, bs, ep, verbose)
    return cnn


def main(verbose=1):
    fp = h5.File(os.path.join('.', 'testdata03.h5'), 'r')
    x = fp['/train/data'].value
    y = fp['/train/label'].value
    num_test = int(fp['/test/num'].value)
    test_x = []
    test_label = []
    test_w = []
    for i in range(num_test):
        test_x.append(fp['/test/data{}'.format(i)].value)
        test_label.append(fp['/test/label{}'.format(i)].value)
        test_w.append(fp['/test/wave{}'.format(i)].value)
    fp.close()

    hid_lr = 0.3
    hid_mt = 0.5
    hid_wd = 0.0001
    ep = 100
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
    }
    rnn = Rn.Rnn(hid_type, hid_param, hid_size, out_type, out_param)
    rnn.train(x, y, ep, window, stride, verbose)

    ann = train_mlp(x, y, verbose)

    for idx in range(num_test):
        p = rnn.simulate(test_x[idx])
        a = ann.simulate(test_x[idx])
        p_fmt = np.argmax(p, axis=1)
        a_fmt = np.argmax(a, axis=1)
        y_fmt = np.argmax(test_label[idx], axis=1)
        print('')
        print(p_fmt)
        print(a_fmt)
        print(y_fmt)
        err_num = np.count_nonzero(y_fmt - p_fmt)
        total_num = p_fmt.size
        print('RNN Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))
        err_num = np.count_nonzero(y_fmt - a_fmt)
        total_num = a_fmt.size
        print('MLP Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))

        kana = ['a', 'i', 'u', 'e', 'o']
        iro = ['#ff0000', '#ffff00', '#ff00ff', '#00ff00', '#00ffff', '#0000ff',
               '#770000', '#007700', '#000077', '#aaff00', '#000000']
        t = np.arange(test_x[idx].shape[0])
        tw = np.arange(test_w[idx].shape[0])
        num_feature = p.shape[1] - 1
        plt.subplot(5, 1, 1)
        plt.plot(tw, test_w[idx])
        plt.subplot(5, 1, 2)
        plt.plot(t, test_x[idx])
        plt.subplot(5, 1, 3)
        for i in range(num_feature):
            plt.plot(t, test_label[idx][:, i], label='{}'.format(kana[i]), color='{}'.format(iro[i]))
        plt.legend(prop={'size': 7})
        plt.ylim(-0.05, 1.1)
        plt.subplot(5, 1, 4)
        for i in range(num_feature):
            plt.plot(t, p[:, i], label='{}'.format(kana[i]), color='{}'.format(iro[i]))
        plt.legend(prop={'size': 7})
        plt.subplot(5, 1, 5)
        for i in range(num_feature):
            plt.plot(t, a[:, i], label='{}'.format(kana[i]), color='{}'.format(iro[i]))
        plt.legend(prop={'size': 7})
        plt.show()


if __name__ == "__main__":
    main()

