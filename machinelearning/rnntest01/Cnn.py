#!/usr/bin/env python

from __future__ import print_function
import NeuralNetwork as Nn
import numpy as np


class Cnn:
    def __init__(self, layer_types=None, hyper_params=None):
        self.layer_types = layer_types if layer_types is not None else [Nn.LAYER_TYPE_DUMMY]
        self.layer_params = hyper_params if hyper_params is not None else [{}]
        self.layer_num = len(self.layer_types)
        self.layer = []

    def layer_train_r(self, x, y, r):
        if r == len(self.layer):
            self.layer.append(
                Nn.create_layer(self.layer_types[r],
                                self.layer_params[r]))
            self.layer[r].initialize_params(x, self.layer_params[r])
        p = self.layer[r].simulate(x)
        if r == self.layer_num - 1:
            delta = y - p  # Assume the cost is the cross-entropy
        else:
            dedy = self.layer_train_r(p, y, r+1)
            delta = self.layer[r].get_delta(p, dedy)
        dedx = self.layer[r].get_dedx(delta)
        self.layer[r].train_with_delta(x, delta)
        return dedx

    def train(self, x, y,
              batch_size=10, epoch=50, verbose=0):
        indices, num_batch = Nn.get_batch_indices(x, batch_size=batch_size)
        for ep in range(epoch):
            if verbose >= 1:
                print('\r ep={}'.format(ep), end='')
            for i in range(num_batch):
                mini_batch = x[indices == i]
                mini_target = y[indices == i]
                self.layer_train_r(mini_batch, mini_target, 0)

    def simulate(self, x2, batch_size=0):
        indices, num_batch = Nn.get_batch_indices(x2, batch_size=batch_size)
        ans = None
        for i in range(num_batch):
            instances = x2[indices == i]
            for r in range(self.layer_num):
                instances = self.layer[r].simulate(instances)
            if ans is None:
                ans_shape = (x2.shape[0],) + instances.shape[1:]
                ans = np.zeros(ans_shape)
            ans[indices == i] = instances
        return ans

    def simulate_getting_all_layers(self, x2, batch_size=0):
        indices, num_batch = Nn.get_batch_indices(x2, batch_size=batch_size)
        all_ans = []
        for i in range(self.layer_num):
            all_ans.append(None)
        for i in range(num_batch):
            instances = x2[indices == i]
            for r in range(self.layer_num):
                instances = self.layer[r].simulate(instances)
                if all_ans[r] is None:
                    ans_shape = (x2.shape[0],) + instances.shape[1:]
                    all_ans[r] = np.zeros(ans_shape)
                (all_ans[r])[indices == i] = instances
        return all_ans


def __generate_sample_image(idx):
    images = np.array([
        [   # Type 0
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [   # Type 1
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ],
        [   # Type 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        [   # Type 3
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        ],
        [   # Type 4
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    ], dtype=np.float)
    ans = images[idx]
    slide = np.random.randint(-3, 4)
    if slide < 0:
        ans = np.append(ans, np.zeros((10, -slide)), axis=1)
        ans = np.delete(ans, range(-slide), axis=1)
    elif slide > 0:
        ans = np.delete(ans, range(10-slide, 10), axis=1)
        ans = np.append(np.zeros((10, slide)), ans, axis=1)
    slide = np.random.randint(-3, 4)
    if slide < 0:
        ans = np.append(ans, np.zeros((-slide, 10)), axis=0)
        ans = np.delete(ans, range(-slide), axis=0)
    elif slide > 0:
        ans = np.delete(ans, range(10-slide, 10), axis=0)
        ans = np.append(np.zeros((slide, 10)), ans, axis=0)
    ans += np.random.normal(scale=0.2, size=(10, 10))
    ans = np.clip(ans, 0, 1)
    return ans


def __train_and_test(verbose=1):
    # Common hyper parameters
    cl_lr = 0.1
    cl_mt = 0.5
    cl_wd = 0.001
    sg_lr = 0.1
    sg_mt = 0.5
    sg_wd = 0.001
    bs = 50
    ep = 80

    # Generate sample data
    num_types = 5
    num_samples = 1000
    num_tests = 500
    x = np.zeros((num_samples, 1, 10, 10))
    y = np.zeros((num_samples, num_types))
    for i in range(num_samples):
        j = np.random.randint(0, num_types)
        y[i, j] = 1
        x[i, 0] = __generate_sample_image(j)
    test_x = np.zeros((num_tests, 1, 10, 10))
    test_y = np.zeros((num_tests, num_types))
    for i in range(num_tests):
        j = np.random.randint(0, num_types)
        test_y[i, j] = 1
        test_x[i, 0] = __generate_sample_image(j)

    # Create model
    types = []
    params = []
    types.append(Nn.LAYER_TYPE_CONVOLUTION)
    params.append({
        'weight_row_size': 3, 'weight_col_size': 3, 'output_feature_num': 10,
        'learning_rate': cl_lr, 'momentum': cl_mt, 'weight_decay': cl_wd
    })
    types.append(Nn.LAYER_TYPE_MAX_POOLING)
    params.append({
        'scale_row_ratio': 2, 'scale_col_ratio': 2
    })
    types.append(Nn.LAYER_TYPE_CONVOLUTION)
    params.append({
        'weight_row_size': 3, 'weight_col_size': 3, 'output_feature_num': 10,
        'learning_rate': cl_lr, 'momentum': cl_mt, 'weight_decay': cl_wd
    })
    types.append(Nn.LAYER_TYPE_MAX_POOLING)
    params.append({
        'scale_row_ratio': 2, 'scale_col_ratio': 2
    })
    types.append(Nn.LAYER_TYPE_CON2SIG)
    params.append({
    })
    types.append(Nn.LAYER_TYPE_SIGMOID)
    params.append({
        'output_size': y.shape[1],
        'learning_rate': sg_lr, 'momentum': sg_mt, 'weight_decay': sg_wd
    })
    cnn = Cnn(types, params)

    # Training
    cnn.train(x, y, bs, ep, verbose)

    # Evaluation
    p = cnn.simulate(test_x, bs)
    p_fmt = np.argmax(p, axis=1)
    t_fmt = np.argmax(test_y, axis=1)
    err_num = np.count_nonzero(t_fmt - p_fmt)
    total_num = p_fmt.size

    print('')
    print(p_fmt)
    print(t_fmt)
    print('Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))


if __name__ == "__main__":
    __train_and_test()

