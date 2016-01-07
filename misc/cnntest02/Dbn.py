#!/usr/bin/env python

from __future__ import print_function
import SigmoidLayer as sl
import SigmoidOutputLayer as so
import NeuralLayer as nl
import numpy as np
import numpy.random as rnd


class Dbn:
    def __init__(self):
        self.hidden = np.array([])
        self.layer_num = 1
        self.layer = [nl.NeuralLayer(0, 0)]

    def get_array_argument(self, arg, size, default=0., dtype=float):
        if arg is None:
            return np.full((size,), default, dtype=dtype)
        if isinstance(arg, float) or isinstance(arg, int):
            return np.full((size,), arg, dtype=dtype)
        list_arg = np.zeros((size,))
        if isinstance(arg, np.array):
            list_arg = arg
        if isinstance(arg, list):
            list_arg = np.array(arg)
        if list_arg.size == size:
            return list_arg
        arg_size = list_arg.size
        if arg_size >= size:
            return list_arg[:size]
        for i in range(size - arg_size):
            list_arg = np.append(list_arg, 0.)
        return list_arg

    def local_layer_pre_train(self, layer, x, y=None, batch_size=0, epoch=50, verbose=0):
        x_size = x.shape[0]
        if batch_size < 1:
            batch_size = x_size
        vint = np.vectorize(lambda q: int(q))
        indices = np.arange(0, x_size)
        indices = vint(indices / batch_size)
        num_batch = np.max(indices) + 1
        for ep in range(epoch):
            if verbose >= 1:
                print('\r ep={}'.format(ep), end='')
            for i in range(num_batch):
                mini_batch = x[indices == i, :]
                if y is None:
                    layer.train_unsupervised(mini_batch)
                else:
                    mini_target = y[indices == i, :]
                    delta = mini_target - layer.simulate(mini_batch)
                    layer.train_with_delta(mini_batch, delta)

    def all_layer_fine_tune(self, x, y, batch_size=0, epoch=50, verbose=0):
        x_size = x.shape[0]
        if batch_size < 1:
            batch_size = x_size
        vint = np.vectorize(lambda q: int(q))
        indices = np.arange(0, x_size)
        indices = vint(indices / batch_size)
        num_batch = np.max(indices) + 1
        for ep in range(epoch):
            if verbose >= 1:
                print('\r ep={}'.format(ep), end='')
            for i in range(num_batch):
                mini_batch = x[indices == i, :]
                mini_target = y[indices == i, :]
                self.local_layer_fine_tune(mini_batch, mini_target, 0)

    def local_layer_fine_tune(self, x, y, r):
        if r == self.layer_num:
            return y - x
        layer = self.layer[r]
        p = layer.simulate(x)
        out_delta = self.local_layer_fine_tune(p, y, r+1)
        in_delta = layer.get_input_delta(x, p, out_delta)
        layer.train_with_delta(x, out_delta)
        return in_delta

    def pre_train(self, x, y, hidden_sizes=[1],
                  learning_rate=0.1, momentum=0.5, weight_decay=0.001,
                  batch_size=10, epoch=50, verbose=0):
        self.hidden = np.array(hidden_sizes)
        self.layer_num = self.hidden.size + 1
        lr = self.get_array_argument(learning_rate, self.layer_num, default=0.1)
        mt = self.get_array_argument(momentum, self.layer_num)
        wd = self.get_array_argument(weight_decay, self.layer_num)
        bs = self.get_array_argument(batch_size, self.layer_num, dtype=int)
        ep = self.get_array_argument(epoch, self.layer_num, dtype=int)
        self.layer = []
        y_col_size = y.shape[1]

        instances = x
        for r in range(self.layer_num - 1):
            n = self.hidden[r]
            sig_layer = sl.SigmoidLayer(n, lr[r], mt[r], wd[r])
            self.layer.append(sig_layer)
            sig_layer.initialize_params(instances, {})
            if verbose >= 1:
                print('\nLayer{} Pre-train'.format(r))
            self.local_layer_pre_train(sig_layer, instances, y=None,
                                       batch_size=bs[r], epoch=ep[r], verbose=verbose)
            instances = sig_layer.simulate(instances)
        r = self.layer_num - 1
        sig_layer = so.SigmoidOutputLayer(y_col_size, lr[r], mt[r], wd[r])
        sig_layer.initialize_params(instances, {})
        self.layer.append(sig_layer)
        if verbose >= 1:
            print('\nLayer{} Pre-train (Supervised)'.format(r))
        self.local_layer_pre_train(sig_layer, instances, y,
                                   batch_size=bs[r], epoch=ep[r], verbose=verbose)

    def fine_tune(self, x, y,
                  learning_rate=0.1, momentum=0.5, weight_decay=0.001,
                  batch_size=10, epoch=50, verbose=0):
        lr = self.get_array_argument(learning_rate, self.layer_num, default=0.1)
        mt = self.get_array_argument(momentum, self.layer_num)
        wd = self.get_array_argument(weight_decay, self.layer_num)
        bs = batch_size
        ep = epoch
        for r in range(self.layer_num):
            self.layer[r].set_hyper_params(learning_rate=lr[r], momentum=mt[r], weight_decay=wd[r])
        if verbose >= 1:
            print('\nFine Tune')
        self.all_layer_fine_tune(x, y, batch_size=bs, epoch=ep, verbose=verbose)

    def simulate(self, x2):
        instances = x2
        for r in range(self.layer_num):
            instances = self.layer[r].simulate(instances)
        return instances


def main(verbose=1):
    data = np.array([[1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1],
                     [1, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 1, 1]])
    label = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1]])
    tsdt = np.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1],
                     [1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 0, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0]])
    tslbl = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0]])

    dbn = Dbn()

    # For Pre-training
    learning_rate = 0.4
    epoch = 1000
    momentum = 0.7
    weight_decay = 0.001
    batch_size = 0
    dbn.pre_train(data, label, hidden_sizes=[4, 4],
                  learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                  batch_size=batch_size, epoch=epoch, verbose=verbose)

    # For Fine-tuning
    learning_rate = 0.075
    epoch = 300
    momentum = 0.5
    weight_decay = 0.001
    batch_size = 0
    dbn.fine_tune(data, label,
                  learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay,
                  batch_size=batch_size, epoch=epoch, verbose=verbose)

    p = dbn.simulate(tsdt)
    p_fmt = np.argmax(p, axis=1)
    t_fmt = np.argmax(tslbl, axis=1)
    err_num = np.count_nonzero(t_fmt - p_fmt)
    total_num = p_fmt.size

    print('')
    print(p_fmt)
    print(t_fmt)
    print('Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))


if __name__ == "__main__":
    main()

