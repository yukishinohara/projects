#!/usr/bin/env python

from __future__ import print_function
import NeuralNetwork as Nn
import numpy as np


class Lstm:
    def __init__(self,
                 hidden_layer_type=None, hidden_layer_param=None, hidden_size=1,
                 out_layer_type=None, out_layer_param=None,
                 input_gate_type=None, input_gate_param=None, input_gate_size=1,
                 output_gate_type=None, output_gate_param=None, output_gate_size=1,
                 forget_gate_type=None, forget_gate_param=None, forget_gate_size=1):
        self.hidden_type = hidden_layer_type
        self.hidden_param = hidden_layer_param
        self.hidden_size = hidden_size
        self.out_type = out_layer_type
        self.out_param = out_layer_param
        self.input_gate_type = input_gate_type
        self.input_gate_param = input_gate_param
        self.input_gate_size = input_gate_size
        self.output_gate_type = output_gate_type
        self.output_gate_param = output_gate_param
        self.output_gate_size = output_gate_size
        self.forget_gate_type = forget_gate_type
        self.forget_gate_param = forget_gate_param
        self.forget_gate_size = forget_gate_size
        self.hidden_layer = None
        self.input_gate = None
        self.output_gate = None
        self.forget_gate = None
        self.out_layer = None

    def train_mini_batch(self, x, y):
        max_time, nx = x.shape
        nh = self.hidden_size
        h = np.zeros((max_time, nh))
        z = np.zeros((max_time, nh))
        i = np.zeros((max_time, nh))
        o = np.zeros((max_time, nh))
        f = np.zeros((max_time, nh))
        c = np.zeros((max_time, nh))
        delta_y = np.zeros(y.shape)
        delta_z = np.zeros((max_time, nh))
        delta_i = np.zeros((max_time, nh))
        delta_o = np.zeros((max_time, nh))
        delta_f = np.zeros((max_time, nh))

        # Feed-forward path
        for t in range(max_time):
            h_prev = h[t-1:t, :] if t != 0 else h[-1:, :]
            c_prev = c[t-1:t, :] if t != 0 else c[-1:, :]
            v_t = np.concatenate((h_prev, x[t:t+1, :]), axis=1)
            z[t:t+1, :] = self.hidden_layer.simulate(v_t)
            f[t:t+1, :] = self.forget_gate.simulate(v_t)
            i[t:t+1, :] = self.input_gate.simulate(v_t)
            o[t:t+1, :] = self.output_gate.simulate(v_t)
            c[t:t+1, :] = (c_prev * f[t:t+1, :]) + (z[t:t+1, :] * o[t:t+1, :])
            h[t:t+1, :] = o[t:t+1, :] * np.tanh(c[t:t+1, :])
            delta_y[t:t+1, :] = self.out_layer.simulate(h[t:t+1, :]) - y[t:t+1, :]

        # Back-propagate path
        dedh_from_future = None
        dedc_from_future = None
        for t in reversed(range(max_time)):
            delta_h_t = self.out_layer.get_dedx(delta_y[t:t+1, :])  # = de/du
            if dedh_from_future is not None:
                delta_h_t += dedh_from_future
            ones = np.ones(c[t:t+1, :].shape)
            tanh_c = np.tanh(c[t:t+1, :])
            delta_c_t = (delta_h_t * o[t:t+1, :] * (ones - tanh_c*tanh_c))
            if dedc_from_future is not None:
                delta_c_t += dedc_from_future
            dedo = delta_h_t * tanh_c
            delta_o[t:t+1, :] = self.output_gate.get_delta(o[t:t+1, :], dedo)
            dedz = delta_c_t * i[t:t+1, :]
            delta_z[t:t+1, :] = self.hidden_layer.get_delta(z[t:t+1, :], dedz)
            dedi = delta_c_t * z[t:t+1, :]
            delta_i[t:t+1, :] = self.input_gate.get_delta(i[t:t+1, :], dedi)
            dedf = delta_c_t
            if t > 0:
                dedf *= c[t-1:t, :]
            delta_f[t:t+1, :] = self.forget_gate.get_delta(f[t:t+1, :], dedf)
            dedc_from_future = f[t:t+1] * delta_c_t
            dedh_from_future = (
                self.output_gate.get_dedx(delta_o[t:t+1, :]) +
                self.hidden_layer.get_dedx(delta_z[t:t+1, :]) +
                self.input_gate.get_dedx(delta_i[t:t+1, :]) +
                self.forget_gate.get_dedx(delta_f[t:t+1, :])
            )[:, nh]

        # Train all layers
        self.out_layer.train_with_delta(h, delta_y)
        v = np.concatenate((h, x), axis=1)
        self.hidden_layer.train_with_delta(v, delta_z)
        self.input_gate.train_with_delta(v, delta_i)
        self.output_gate.train_with_delta(v, delta_o)
        self.forget_gate.train_with_delta(v, delta_f)

    def train(self, x, y,
              epoch=50, window_size=0, stride_size=0, verbose=0):
        max_time, nx = x.shape
        if window_size == 0:
            window_size = max_time
        if stride_size == 0:
            stride_size = max_time
        nh = self.hidden_size
        v_t = np.random.random((1, nx + nh))
        if self.hidden_layer is None:
            self.hidden_layer = Nn.create_layer(self.hidden_type, self.hidden_param)
            self.hidden_layer.initialize_params(v_t, self.hidden_param)
        if self.input_gate is None:
            self.input_gate = Nn.create_layer(self.input_gate_type, self.input_gate_param)
            self.input_gate.initialize_params(v_t, self.input_gate_param)
        if self.output_gate is None:
            self.output_gate = Nn.create_layer(self.output_gate_type, self.output_gate_param)
            self.output_gate.initialize_params(v_t, self.output_gate_param)
        if self.forget_gate is None:
            self.forget_gate = Nn.create_layer(self.forget_gate_type, self.forget_gate_param)
            self.forget_gate.initialize_params(v_t, self.forget_gate_param)
        if self.out_layer is None:
            u_t = np.random.random((1, nh))
            self.out_layer = Nn.create_layer(self.out_type, self.out_param)
            self.out_layer.initialize_params(u_t, self.out_param)
        for ep in range(epoch):
            if verbose >= 1:
                print('\r ep={}'.format(ep), end='')
            if window_size == max_time and stride_size == max_time:
                self.train_mini_batch(x, y)
            for start_time in range(0, max_time, stride_size):
                end_time = min(max_time, (start_time + window_size))
                self.train_mini_batch(
                    x[start_time:end_time, :],
                    y[start_time:end_time, :]
                )

    def simulate(self, x2):
        max_time = x2.shape[0]
        h_prev = np.zeros((1, self.hidden_size))
        c_prev = np.zeros((1, self.hidden_size))
        y = None
        for t in range(max_time):
            v_t = np.concatenate((h_prev, x2[t:t+1, :]), axis=1)
            z_t = self.hidden_layer.simulate(v_t)
            f_t = self.forget_gate.simulate(v_t)
            i_t = self.input_gate.simulate(v_t)
            o_t = self.output_gate.simulate(v_t)
            c_prev = (c_prev * f_t) + (z_t * i_t)
            h_prev = o_t * np.tanh(c_prev)
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
    hid_lr = 0.1
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
    ep = 100
    window = 0
    stride = 0

    # Generate sample data
    x, y, x_test = __generate_sample_string()

    # Create model
    nh = 26
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
    lstm = Lstm(
        hid_type, hid_param, hid_size,
        out_type, out_param,
        igt_type, igt_param, igt_size,
        ogt_type, ogt_param, ogt_size,
        fgt_type, fgt_param, fgt_size
    )

    # Training
    lstm.train(x, y, ep, window, stride, verbose)

    # Evaluation
    p = lstm.simulate(x_test)
    p_fmt = np.argmax(p, axis=1)
    p_str = ''
    for i in p_fmt:
        p_str += chr(i + ord('a'))
    print('')
    print(p_str)


if __name__ == "__main__":
    __train_and_test()

