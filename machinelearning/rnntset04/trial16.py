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
    test_label = []
    test_w = []
    for i in range(num_test):
        test_x.append(fp['/test/data{}'.format(i)].value)
        if '/test/label{}'.format(i) in fp.keys():
            test_label.append(fp['/test/label{}'.format(i)].value)
        else:
            tmpx = fp['/test/data{}'.format(i)].value
            test_label.append(np.zeros((tmpx.shape[0], y.shape[1])))
        test_w.append(fp['/test/wave{}'.format(i)].value)
    fp.close()

    hid_lr = 0.01
    hid_mt = 0.6
    hid_wd = 0.0001
    ep = 5000
    window = 0
    stride = 0

    hid_type = Nn.LAYER_TYPE_TANH
    hid_size = y.shape[1]
    hid_param = {
        'output_size': hid_size,
        'learning_rate': hid_lr, 'momentum': hid_mt, 'weight_decay': hid_wd
    }
    out_type = Nn.LAYER_TYPE_SOFTMAX
    out_param = {
    }
    rnn = Rn.Rnn(hid_type, hid_param, hid_size, out_type, out_param)
    rnn.train(x, y, ep, window, stride, verbose)

    for idx in range(num_test):
        p = rnn.simulate(test_x[idx])
        p_fmt = np.argmax(p, axis=1)
        y_fmt = np.argmax(test_label[idx], axis=1)
        print('')
        print(p_fmt)
        print(y_fmt)
        err_num = np.count_nonzero(y_fmt - p_fmt)
        total_num = p_fmt.size
        print('RNN Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))

        kana = ['a', 'i', 'u', 'e', 'o', '', '', '', '', '', '', '']
        iro = ['#ff0000', '#ffff00', '#ff00ff', '#00ff00', '#00ffff', '#0000ff',
               '#770000', '#007700', '#000077', '#aaff00', '#00ffaa', '#000000']
        t = np.arange(test_x[idx].shape[0])
        tw = np.arange(test_w[idx].shape[0])
        num_feature = p.shape[1] - 1
        plt.subplot(4, 1, 1)
        plt.plot(tw, test_w[idx])
        plt.subplot(4, 1, 2)
        plt.plot(t, test_x[idx])
        plt.subplot(4, 1, 3)
        for i in range(num_feature):
            plt.plot(t, test_label[idx][:, i], label='{}'.format(kana[i]), color='{}'.format(iro[i]))
        plt.legend(prop={'size': 7})
        plt.ylim(-0.05, 1.1)
        plt.subplot(4, 1, 4)
        for i in range(num_feature):
            plt.plot(t, p[:, i], label='{}'.format(kana[i]), color='{}'.format(iro[i]))
        plt.legend(prop={'size': 7})
        plt.ylim(-0.05, 1.1)
        plt.show()


if __name__ == "__main__":
    main()

