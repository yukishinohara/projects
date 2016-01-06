#!/usr/bin/env python

from __future__ import print_function
import Cnn as Cn


def func(arg0, arg1, arg2=2, arg3=3, arg4='baka'):
    print('arg0={}, arg1={}, arg2={}, arg3={}, arg4={}'.format(
        arg0, arg1, arg2, arg3, arg4
    ))


def main():
    params = [
        {'arg0': 999, 'arg1': 123},
        {'arg3': 999, 'arg1': 123, 'arg0': 12345},
        {'arg0': 999, 'arg1': 123, 'arg2': 12345, 'arg4': 'aho'},
        {'arg0': Cn.LAYER_TYPE_CONVOLUTION, 'arg1': Cn.LAYER_TYPE_MAX_POOLING, 'arg3': 12345, 'arg4': 'aho'},
    ]
    params[2].update({'arg3': 99999, 'arg2': 11111})
    for param in params:
        func(**param)


if __name__ == "__main__":
    main()

