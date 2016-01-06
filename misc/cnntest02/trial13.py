#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import Cnn as Cn
import mnistloader as mn
from PIL import Image


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
    num_samples = 300
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
    types.append(Cn.LAYER_TYPE_CONVOLUTION)
    params.append({
        'weight_row_size': 3, 'weight_col_size': 3, 'output_feature_num': 10,
        'learning_rate': cl_lr, 'momentum': cl_mt, 'weight_decay': cl_wd
    })
    types.append(Cn.LAYER_TYPE_MAX_POOLING)
    params.append({
        'scale_row_ratio': 2, 'scale_col_ratio': 2
    })
    types.append(Cn.LAYER_TYPE_CONVOLUTION)
    params.append({
        'weight_row_size': 3, 'weight_col_size': 3, 'output_feature_num': 10,
        'learning_rate': cl_lr, 'momentum': cl_mt, 'weight_decay': cl_wd
    })
    types.append(Cn.LAYER_TYPE_MAX_POOLING)
    params.append({
        'scale_row_ratio': 2, 'scale_col_ratio': 2
    })
    types.append(Cn.LAYER_TYPE_CON2SIG)
    params.append({
    })
    types.append(Cn.LAYER_TYPE_SIGMOID)
    params.append({
        'output_size': y.shape[1],
        'learning_rate': sg_lr, 'momentum': sg_mt, 'weight_decay': sg_wd
    })
    cnn = Cn.Cnn(types, params)

    # Training
    cnn.train(x, y, bs, ep, verbose)

    # Evaluation
    ps = cnn.simulate_getting_all_layers(test_x, bs)
    p = ps[len(types) - 1]
    p_fmt = np.argmax(p, axis=1)
    t_fmt = np.argmax(test_y, axis=1)
    err_num = np.count_nonzero(t_fmt - p_fmt)
    total_num = p_fmt.size

    print('')
    print(p_fmt)
    print(t_fmt)
    print('Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))

    img_tool = mn.MNISTloader2D()
    for r in range(len(types)):
        if types[r] == Cn.LAYER_TYPE_MAX_POOLING:
            for f in range(10):
                yi = img_tool.render(ps[r], width=37, f=f)
                img1 = Image.fromarray(np.uint8(yi * 255))
                img1.show()
    test_xi = img_tool.render(test_x, width=37)
    img2 = Image.fromarray(np.uint8(test_xi * 255))
    img2.show()


if __name__ == "__main__":
    __train_and_test()


