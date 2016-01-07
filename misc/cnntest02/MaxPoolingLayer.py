#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from numpy.lib.stride_tricks import as_strided
import DummyLayer as Dl
import warnings


class MaxPoolingLayer(Dl.DummyLayer):
    def __init__(self, scale_row_ratio, scale_col_ratio):
        # Parameters
        self.xh = self.xw = self.yh = self.yw = 1
        self.ah = scale_row_ratio
        self.aw = scale_col_ratio
        self.f = 1
        self.m = None
        # m is a matrix representing pixels of units that were selected as max ones, and is used for
        # calculating input_delta in the back-propagation stage.
        # <lifetime of m>
        #  m is determined when the feed-forward process, and keep its value until the training stage.
        #  It is expected that input_delta should be calculated before the training process.
        #   (This is natural because the back-propagation training needs the input_delta to propagate
        #    delta to the next layer before the weights are updated)
        #  In other word, m always keeps the selected pixels for the latest output image unless the
        #  training happens.
        Dl.DummyLayer.__init__(self)

    def initialize_params(self, x, hyper_params):
        _, self.f, self.xh, self.xw = x.shape
        if self.xh % self.ah != 0 or self.xw % self.aw != 0:
            warnings.warn('''
            The image sizes should be multiples of the scale ratios
            ({}, {}) cannot be divided by ({}, {})'''.format(self.xh, self.xw, self.ah, self.aw))
        self.yh = int(self.xh / self.ah)
        self.yw = int(self.xw / self.aw)

    def train_with_delta(self, x, output_delta):
        self.m = None

    def simulate(self, x):
        d = x.shape[0]
        x2 = as_strided(x,
                        shape=(d, self.f, self.yh, self.yw, self.ah, self.aw),
                        strides=x.strides[:2] + (x.strides[2]*self.ah, x.strides[3]*self.aw) + x.strides[-2:])
        y = np.amax(np.amax(x2, axis=5), axis=4)
        y2 = (x2 >= y.reshape(y.shape + (1, 1))) + 0.
        m_count = np.sum(np.sum(y2, axis=5), axis=4)
        m_count3 = np.repeat(np.repeat(m_count, self.aw, axis=3), self.ah, axis=2)
        y3 = np.repeat(np.repeat(y, self.aw, axis=3), self.ah, axis=2)
        self.m = ((y3 == x) + 0.) / m_count3
        return y

    def predict(self, x):
        return self.simulate(x)

    def get_input_delta(self, output_delta):
        dy3 = np.repeat(np.repeat(output_delta, self.aw, axis=3), self.ah, axis=2)
        dx = self.m * dy3
        return dx
