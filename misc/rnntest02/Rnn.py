#!/usr/bin/env python

from __future__ import print_function
import NeuralNetwork as Nn
import numpy as np


class Rnn:
    def __init__(self,
                 hidden_layer_type=None, hidden_layer_param=None, hidden_size=1,
                 out_layer_type=None, out_layer_param=None):
        self.hidden_type = hidden_layer_type
        self.hidden_param = hidden_layer_param
        self.hidden_size = hidden_size
        self.out_type = out_layer_type
        self.out_param = out_layer_param
        self.hidden_layer = None
        self.out_layer = None

    def train_mini_batch(self, x, y):
        max_time, nx = x.shape
        nh = self.hidden_size
        h = np.zeros((max_time, nh))
        delta_out = np.zeros(y.shape)
        delta_hidden = np.zeros((max_time, nh))
        for t in range(max_time):
            h_prev = h[t-1:t, :] if t != 0 else h[-1:, :]
            z_t = np.concatenate((h_prev, x[t:t+1, :]), axis=1)
            h[t:t+1, :] = self.hidden_layer.simulate(z_t)
            delta_out[t:t+1, :] = y[t:t+1, :] - self.out_layer.simulate(h[t:t+1, :])

        dedh_from_future = None
        for t in reversed(range(max_time)):
            dedh = self.out_layer.get_dedx(delta_out[t:t+1, :])
            if dedh_from_future is not None:
                dedh += dedh_from_future
            delta_hidden[t:t+1, :] = self.hidden_layer.get_delta(h[t:t+1, :], dedh)
            dedz = self.hidden_layer.get_dedx(delta_hidden[t:t+1, :])
            dedh_from_future = dedz[:, :nh]

        self.out_layer.train_with_delta(h, delta_out)
        self.hidden_layer.train_with_delta(np.concatenate((h, x), axis=1), delta_hidden)

    def train(self, x, y,
              epoch=50, window_size=0, stride_size=0, verbose=0):
        max_time, nx = x.shape
        if window_size == 0:
            window_size = max_time
        if stride_size == 0:
            stride_size = max_time
        nh = self.hidden_size
        z_t = np.random.random((1, nx + nh))
        if self.hidden_layer is None:
            self.hidden_layer = Nn.create_layer(self.hidden_type, self.hidden_param)
            self.hidden_layer.initialize_params(z_t, self.hidden_param)
        if self.out_layer is None:
            h_t = np.random.random((1, nh))
            self.out_layer = Nn.create_layer(self.out_type, self.out_param)
            self.out_layer.initialize_params(h_t, self.out_param)
        for ep in range(epoch):
            if verbose >= 1:
                print('\r ep={}'.format(ep), end='')
            for start_time in range(0, max_time, stride_size):
                end_time = min(max_time, (start_time + window_size))
                self.train_mini_batch(
                    x[start_time:end_time, :],
                    y[start_time:end_time, :]
                )

    def simulate(self, x2):
        max_time = x2.shape[0]
        h_prev = np.zeros((1, self.hidden_size))
        y = None
        for t in range(max_time):
            z_t = np.concatenate((h_prev, x2[t:t+1, :]), axis=1)
            h_prev = self.hidden_layer.simulate(z_t)
            if y is None:
                y = self.out_layer.simulate(h_prev)
            else:
                y = np.append(y, self.out_layer.simulate(h_prev), axis=0)
        return y


def __generate_sample_string():
    x_str = "ppxxqqxxrrxx"
    y_str = "aaaabbbbcccc"
    x_test_str = "xxqqxxrrxxppxxqqxxxxrxpxqxprxqpx"
    rep = 3
    str_len = len(x_str)
    max_time = str_len * rep
    x = np.zeros((max_time, 26), dtype=np.int)
    y = np.zeros((max_time, 26), dtype=np.int)
    x_test = np.zeros((len(x_test_str), 26), dtype=np.int)
    for i in range(rep):
        for j in range(str_len):
            ac = ord(x_str[j]) - ord('a')
            x[i*str_len + j, ac] = 1
            ac = ord(y_str[j]) - ord('a')
            y[i*str_len + j, ac] = 1
    for j in range(len(x_test_str)):
        ac = ord(x_test_str[j]) - ord('a')
        x_test[j, ac] = 1
    return x, y, x_test


def __train_and_test(verbose=1):
    # Common hyper parameters
    hid_lr = 0.5
    hid_mt = 0.5
    hid_wd = 0.0001
    ep = 100
    window = 0
    stride = 0

    # Generate sample data
    x, y, x_test = __generate_sample_string()

    # Create model
    hid_type = Nn.LAYER_TYPE_SIGMOID
    hid_size = 26
    hid_param = {
        'output_size': hid_size,
        'learning_rate': hid_lr, 'momentum': hid_mt, 'weight_decay': hid_wd
    }
    out_type = Nn.LAYER_TYPE_IDENTITY
    out_param = {
    }
    rnn = Rnn(hid_type, hid_param, hid_size, out_type, out_param)

    # Training
    rnn.train(x, y, ep, window, stride, verbose)

    # Evaluation
    p = rnn.simulate(x_test)
    p_fmt = np.argmax(p, axis=1)
    p_str = ''
    for i in p_fmt:
        p_str += chr(i + ord('a'))
    print('')
    print(p_str)


if __name__ == "__main__":
    __train_and_test()

