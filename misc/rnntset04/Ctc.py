#!/usr/bin/env python

from __future__ import print_function
import NeuralNetwork as Nn
import numpy as np


class Ctc:
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

    @staticmethod
    def generate_label_sequence(label):
        char2index = {}
        for c in label:
            char2index[c] = 1
        keys = char2index.keys()
        char2index = {}
        index2char = []
        for c in keys:
            char2index[c] = len(index2char)
            index2char.append(c)
        index2char.append('_')
        num_stat = len(index2char)
        seq_y = []
        for c in label:
            seq_y.append(char2index[c])
        return num_stat, num_stat-1, seq_y, index2char

    def train_mini_batch(self, x, size_y, seq_y, blank):
        max_time, nx = x.shape
        max_stat = 2 * len(seq_y) + 1
        nh = self.hidden_size
        h = np.zeros((max_time, nh))
        p = np.zeros((max_time, size_y))
        delta_out = np.zeros(p.shape)
        delta_hidden = np.zeros((max_time, nh))

        # Forward
        for t in range(max_time):
            h_prev = h[t-1:t, :] if t != 0 else h[-1:, :]
            z_t = np.concatenate((h_prev, x[t:t+1, :]), axis=1)
            h[t:t+1, :] = self.hidden_layer.simulate(z_t)
            p[t:t+1, :] = self.out_layer.simulate(h[t:t+1, :])

        # Alpha and Beta
        alpha = np.zeros((max_time, max_stat))
        beta = np.zeros((max_time, max_stat))
        alpha[0, 0] = p[0, blank]
        alpha[0, 1] = p[0, seq_y[0]]
        for t in range(1, max_time):
            start = max(0, max_stat - 2*(max_time - t))
            end = min(2*t + 2, max_stat)
            for s in range(start, max_stat):
                label = int(int(s)/ 2)
                if s % 2 == 0:
                    if s == 0:
                        alpha[t, s] = alpha[t-1, s] * p[t, blank]
                    else:
                        alpha[t, s] = (alpha[t-1, s] + alpha[t-1, s-1]) * p[t, blank]
                else:
                    if s == 1:
                        alpha[t, s] = (alpha[t-1, s] + alpha[t-1, s-1]) * p[t, seq_y[label]]
                    elif label > 0 and seq_y[label] == seq_y[label - 1]:
                        alpha[t, s] = (alpha[t-1, s] + alpha[t-1, s-1]) * p[t, seq_y[label]]
                    else:
                        alpha[t, s] = (alpha[t-1, s] + alpha[t-1, s-1] + alpha[t-1, s-2]) * p[t, seq_y[label]]
        beta[-1, -1] = p[-1, blank]
        beta[-1, -2] = p[-1, seq_y[-1]]
        for t in reversed(range(0, max_time-1)):
            start = max(0, max_stat - 2*(max_time-t))
            end = min(2*t + 2, max_stat)
            for s in reversed(range(0, end)):
                label = int(int(s)/ 2)
                if s % 2 == 0:
                    if s == max_stat - 1:
                        beta[t, s] = beta[t+1, s] * p[t, blank]
                    else:
                        beta[t, s] = (beta[t+1, s] + beta[t+1, s+1]) * p[t, blank]
                else:
                    if s == max_stat - 2:
                        beta[t, s] = (beta[t+1, s] + beta[t+1, s+1]) * p[t, seq_y[label]]
                    elif label < max_stat-1 and seq_y[label] == seq_y[label + 1]:
                        beta[t, s] = (beta[t+1, s] + beta[t+1, s+1]) * p[t, seq_y[label]]
                    else:
                        beta[t, s] = (beta[t+1, s] + beta[t+1, s+1] + beta[t+1, s+2]) * p[t, seq_y[label]]
        ab = alpha * beta
        for s in range(max_stat):
            label = int(int(s)/ 2)
            if s % 2 == 0:
                delta_out[:, blank] += ab[:, s]
            else:
                delta_out[:, seq_y[label]] += ab[:, s]
        ab_sum = np.sum(ab, axis=1).reshape((max_time, 1))
        delta_out = (delta_out - p) / (p * ab_sum)

        # Backward
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

    def train(self, x, label,
              epoch=50, window_size=0, stride_size=0, verbose=0):
        max_time, nx = x.shape
        size_y, blank, seq_y, char2index = self.generate_label_sequence(label)
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
            if window_size == max_time and stride_size == max_time:
                self.train_mini_batch(x, size_y, seq_y, blank)
            for start_time in range(0, max_time, stride_size):
                end_time = min(max_time, (start_time + window_size))
                self.train_mini_batch(
                    x[start_time:end_time, :],
                    size_y, seq_y, blank
                )
        return char2index

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
    rnn = Ctc(hid_type, hid_param, hid_size, out_type, out_param)

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

