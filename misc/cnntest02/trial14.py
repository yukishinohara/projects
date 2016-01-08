#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import Cnn as Cn
import mnistloader as mn
from PIL import Image
import warnings
import shelve
import os
import zipfile

warnings.filterwarnings('error')


def __save_model(cnn, bs):
    directory = os.path.join('.', '__trained__')
    if not os.path.exists(directory):
        os.makedirs(directory)
    datafile = shelve.open(os.path.join(directory, 'cnn_mnist'))
    datafile['cnn'] = cnn
    datafile['bs'] = bs
    datafile.close()
    zf = zipfile.ZipFile('cnn_mnist.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(os.path.join('.', '__trained__')):
        for file in files:
            zf.write(os.path.join(root, file))
    zf.close()


def __test(cnn, test_x, test_y, bs):
    p = cnn.simulate(test_x, bs)
    p_fmt = np.argmax(p, axis=1)
    t_fmt = np.argmax(test_y, axis=1)
    err_num = np.count_nonzero(t_fmt - p_fmt)
    total_num = p_fmt.size
    print('')
    print(p_fmt)
    print(t_fmt)
    print('Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))


def __tmp_show_images(cnn, test_x, bs):
    ps = cnn.simulate_getting_all_layers(test_x, bs)
    img_tool = mn.MNISTloader2D('__data__')
    for f in range(16):
        yi = img_tool.render(ps[0], width=37, f=f)
        img1 = Image.fromarray(np.uint8(yi * 255))
        img1.show()
    test_xi = img_tool.render(test_x, width=37)
    img2 = Image.fromarray(np.uint8(test_xi * 255))
    img2.show()


def __load_model_and_test(test_size=1000):
    # Load data and model
    loader = mn.MNISTloader2D('__data__')
    [test_x, test_y] = loader.load_test_data(msize=test_size)
    zf = zipfile.ZipFile('cnn_mnist.zip', 'r')
    zf.extractall('.')
    zf.close()
    datafile = shelve.open(os.path.join('.', '__trained__', 'cnn_mnist'))
    cnn = datafile['cnn']
    bs = datafile['bs']
    datafile.close()

    # Test
    __test(cnn, test_x, test_y, bs)


def __train_and_test(verbose=2):
    # Load data
    loader = mn.MNISTloader2D('__data__')
    [x, y] = loader.load_train_data(msize=7000)
    [test_x, test_y] = loader.load_test_data(msize=1000)

    # Common hyper parameters
    cl_lr = 0.02
    cl_mt = 0.6
    cl_wd = 0.001
    sg_lr = 0.03
    sg_mt = 0.6
    sg_wd = 0.001
    bs = 20
    ep = 5
    ep_repeat = 30

    # Create model
    types = []
    params = []
    #     Convolution    MaxPool  Convolution  Convolution   MaxPool   Convolution    Squeeze     Logistic
    # (28, 28) -> (24, 24) -> (12, 12) ->  (8, 8)  ->  (6, 6)  ->   (3, 3)  ->  (1, 1)  ->  (120,)  ->  (10,)
    types.append(Cn.LAYER_TYPE_CONVOLUTION)
    params.append({
        'weight_row_size': 5, 'weight_col_size': 5, 'output_feature_num': 8,
        'learning_rate': cl_lr, 'momentum': cl_mt, 'weight_decay': cl_wd
    })
    types.append(Cn.LAYER_TYPE_MAX_POOLING)
    params.append({
        'scale_row_ratio': 2, 'scale_col_ratio': 2
    })
    types.append(Cn.LAYER_TYPE_CONVOLUTION)
    params.append({
        'weight_row_size': 5, 'weight_col_size': 5, 'output_feature_num': 12,
        'learning_rate': cl_lr, 'momentum': cl_mt, 'weight_decay': cl_wd
    })
    types.append(Cn.LAYER_TYPE_CONVOLUTION)
    params.append({
        'weight_row_size': 3, 'weight_col_size': 3, 'output_feature_num': 16,
        'learning_rate': cl_lr, 'momentum': cl_mt, 'weight_decay': cl_wd
    })
    types.append(Cn.LAYER_TYPE_MAX_POOLING)
    params.append({
        'scale_row_ratio': 2, 'scale_col_ratio': 2
    })
    types.append(Cn.LAYER_TYPE_CONVOLUTION)
    params.append({
        'weight_row_size': 3, 'weight_col_size': 3, 'output_feature_num': 120,
        'learning_rate': cl_lr, 'momentum': cl_mt, 'weight_decay': cl_wd
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

    # Training with intermediate reports
    for epr in range(ep_repeat):
        if verbose >= 1:
            print ('Loop {}/{}'.format(epr, ep_repeat))
        cnn.train(x, y, bs, ep, verbose)
        __test(cnn, test_x, test_y, bs)
        __save_model(cnn, bs)


if __name__ == "__main__":
    # __load_model_and_test()
    __train_and_test()

