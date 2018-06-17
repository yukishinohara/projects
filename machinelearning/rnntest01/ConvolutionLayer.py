#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from numpy.lib.stride_tricks import as_strided
import DummyLayer as Dl


class ConvolutionLayer(Dl.DummyLayer):
    def __init__(self, weight_row_size, weight_col_size, output_feature_num,
                 learning_rate=0.1, momentum=0.5, weight_decay=0.001):
        # Parameters
        self.xh = self.xw = self.yh = self.yw = 1
        self.wh = weight_row_size
        self.ww = weight_col_size
        self.f = 1
        self.g = output_feature_num
        self.lr = learning_rate
        self.mt = momentum
        self.wd = weight_decay
        self.c = 0.
        self.w = 0.
        # For the momentum term
        self.delta_w = 0.
        self.delta_c = 0.
        Dl.DummyLayer.__init__(self)

    def initialize_params(self, x, hyper_params):
        _, self.f, self.xh, self.xw = x.shape
        self.yh = self.xh - self.wh + 1
        self.yw = self.xw - self.ww + 1
        self.w = np.zeros((self.f, self.g, self.wh, self.ww))
        self.c = np.zeros((self.g, self.yh, self.yw))
        # Max bound from http://www.deeplearning.net/tutorial/lenet.html
        bound = np.sqrt(6. / (self.wh * self.ww * (self.f + self.g)))
        self.w += np.random.uniform(-bound, bound, size=self.w.shape)
        self.set_hyper_params(**hyper_params)
        return

    def update_hyper_params(self, hyper_params):
        self.set_hyper_params(**hyper_params)

    def set_hyper_params(self, learning_rate=None, momentum=None, weight_decay=None, **ignored):
        if learning_rate is not None:
            self.lr = learning_rate
        if momentum is not None:
            self.mt = momentum
        if weight_decay is not None:
            self.wd = weight_decay

    def train_unsupervised(self, x):
        pass

    def train_with_delta(self, x, delta):
        d = x.shape[0]
        x3 = as_strided(x,
                        shape=(d, self.f, self.wh, self.ww, self.yh, self.yw),
                        strides=x.strides+x.strides[-2::])
        dw = np.einsum('ukrs,ulijrs->lkij', delta, x3)
        dc = np.sum(delta, axis=0)
        alpha = self.lr / float(d)
        self.delta_w = alpha*dw + self.mt*self.delta_w - self.wd*self.w
        self.delta_c = alpha*dc + self.mt*self.delta_c - self.wd*self.c
        self.w += self.delta_w
        self.c += self.delta_c

    def simulate(self, x):
        d = x.shape[0]
        x2 = as_strided(x,
                        shape=(d, self.f, self.yh, self.yw, self.wh, self.ww),
                        strides=x.strides+x.strides[-2:])
        y = np.einsum('qkrs,lqijrs->lkij', self.w, x2) + self.c
        return y

    def predict(self, x):
        return self.simulate(x)

    def get_delta(self, y, dedy):
        return dedy  # The activation function is f(x) = x

    def get_dedx(self, delta):
        d = delta.shape[0]
        xh = self.yh + self.wh - 1
        xw = self.yw + self.ww - 1
        wf = self.w[:, :, ::-1, ::-1]
        z_row = np.zeros((d, self.g, self.wh - 1, self.yw))
        z_col = np.zeros((d, self.g, xh + self.wh - 1, self.ww - 1))

        dy2 = np.append(np.append(z_col,
                                  np.append(np.append(z_row, delta, axis=2), z_row, axis=2),
                                  axis=3), z_col, axis=3)
        dy3 = as_strided(dy2,
                         shape=(d, self.g, xh, xw, self.wh, self.ww),
                         strides=dy2.strides+dy2.strides[-2::])
        dedx = np.einsum('kprs,lpijrs->lkij', wf, dy3)
        return dedx

