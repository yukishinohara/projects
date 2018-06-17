#!/usr/bin/env python

from __future__ import print_function
import SigmoidLayer as Sl
import DummyLayer as Dl
import ConvolutionLayer as Cl
import MaxPoolingLayer as Ml
import ConvoPool2SigmoidLayer as Il
import IdentityLayer as Tl
import HyperbolicTangentLayer as Hl
import SoftmaxLayer as Xl
import numpy as np


LayerTypes = (
    LAYER_TYPE_DUMMY,
    LAYER_TYPE_SIGMOID,
    LAYER_TYPE_CONVOLUTION,
    LAYER_TYPE_MAX_POOLING,
    LAYER_TYPE_CON2SIG,
    LAYER_TYPE_IDENTITY,
    LAYER_TYPE_TANH,
    LAYER_TYPE_SOFTMAX
) = range(0, 8)


def get_array_argument(arg, size, default=0., dtype=float):
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
        list_arg = np.append(list_arg, np.array([0.]))
    return list_arg


def get_batch_indices(x, batch_size=0):
    x_size = x.shape[0]
    if batch_size < 1:
        batch_size = x_size
    v_int = np.vectorize(lambda q: int(q))
    indices = np.arange(0, x_size)
    indices = v_int(indices / batch_size)
    num_batch = np.max(indices) + 1
    return indices, num_batch


def create_layer(layer_type, hyper_params):
    if layer_type == LAYER_TYPE_SIGMOID:
        return Sl.SigmoidLayer(**hyper_params)
    elif layer_type == LAYER_TYPE_CONVOLUTION:
        return Cl.ConvolutionLayer(**hyper_params)
    elif layer_type == LAYER_TYPE_MAX_POOLING:
        return Ml.MaxPoolingLayer(**hyper_params)
    elif layer_type == LAYER_TYPE_CON2SIG:
        return Il.ConvoPool2SigmoidLayer()
    elif layer_type == LAYER_TYPE_IDENTITY:
        return Tl.IdentityLayer()
    elif layer_type == LAYER_TYPE_TANH:
        return Hl.HyperbolicTangentLayer(**hyper_params)
    elif layer_type == LAYER_TYPE_SOFTMAX:
        return Xl.SoftmaxLayer(**hyper_params)
    else:
        return Dl.DummyLayer()

