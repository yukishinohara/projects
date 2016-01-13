#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import Lstm
import NeuralNetwork as Nn
import shelve
import zipfile


def save_model(rnn):
    directory = os.path.join('.', '__trained__')
    if not os.path.exists(directory):
        os.makedirs(directory)
    datafile = shelve.open(os.path.join(directory, 'lstm_wave'))
    datafile['lstm'] = rnn
    datafile.close()
    zf = zipfile.ZipFile('lstm_wave.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(os.path.join('.', '__trained__')):
        for file in files:
            zf.write(os.path.join(root, file))
    zf.close()


def main(verbose=1):
    fp = h5.File(os.path.join('.', 'testdata02.h5'), 'r')
    x = fp['/train/data'].value
    y = fp['/train/label'].value
    num_test = int(fp['/test/num'].value)
    test_x = []
    test_w = []
    for i in range(num_test):
        test_x.append(fp['/test/data{}'.format(i)].value)
        test_w.append(fp['/test/wave{}'.format(i)].value)
    fp.close()

    # Common hyper parameters
    hid_lr = 0.2
    hid_mt = 0.5
    hid_wd = 0.0001
    igt_lr = 0.1
    igt_mt = 0.5
    igt_wd = 0.0001
    ogt_lr = 0.1
    ogt_mt = 0.5
    ogt_wd = 0.0001
    fgt_lr = 0.1
    fgt_mt = 0.5
    fgt_wd = 0.0001
    out_lr = 0.1
    out_mt = 0.5
    out_wd = 0.0001
    ep = 3
    repeat = 30
    window = 0
    stride = 0

    # Create model
    nh = y.shape[1]
    hid_type = Nn.LAYER_TYPE_TANH
    hid_size = nh
    hid_param = {
        'output_size': hid_size,
        'learning_rate': hid_lr, 'momentum': hid_mt, 'weight_decay': hid_wd
    }
    igt_type = Nn.LAYER_TYPE_SIGMOID
    igt_size = nh
    igt_param = {
        'output_size': igt_size,
        'learning_rate': igt_lr, 'momentum': igt_mt, 'weight_decay': igt_wd
    }
    ogt_type = Nn.LAYER_TYPE_SIGMOID
    ogt_size = nh
    ogt_param = {
        'output_size': ogt_size,
        'learning_rate': ogt_lr, 'momentum': ogt_mt, 'weight_decay': ogt_wd
    }
    fgt_type = Nn.LAYER_TYPE_SIGMOID
    fgt_size = nh
    fgt_param = {
        'output_size': fgt_size,
        'learning_rate': fgt_lr, 'momentum': fgt_mt, 'weight_decay': fgt_wd
    }
    out_type = Nn.LAYER_TYPE_IDENTITY
    out_param = {
    }
    lstm = Lstm.Lstm(
        hid_type, hid_param, hid_size,
        out_type, out_param,
        igt_type, igt_param, igt_size,
        ogt_type, ogt_param, ogt_size,
        fgt_type, fgt_param, fgt_size
    )

    # Training
    for rep in range(repeat):
        if verbose >= 1:
            print('Repeat {} / {}'.format(rep, repeat))
        lstm.train(x, y, ep, window, stride, verbose)
        save_model(lstm)

    for idx in range(num_test):
        p = lstm.simulate(test_x[idx])
        p_fmt = np.argmax(p, axis=1)
        print('')
        print(p_fmt)

        iro = ['#ff0000', '#ffff00', '#ff00ff', '#00ff00', '#00ffff', '#0000ff',
               '#770000', '#007700', '#000077', '#aaff00', '#00aaff', '#000000']
        t = np.arange(test_x[idx].shape[0])
        tw = np.arange(test_w[idx].shape[0])
        num_feature = p.shape[1]
        plt.subplot(3, 1, 1)
        plt.plot(tw, test_w[idx])
        plt.subplot(3, 1, 2)
        plt.plot(t, test_x[idx])
        plt.subplot(3, 1, 3)
        for i in range(num_feature):
            plt.plot(t, p[:, i], label='{}'.format(i), color='{}'.format(iro[i]))
        plt.legend(prop={'size': 7})
        plt.show()


if __name__ == "__main__":
    main()

