#!/usr/bin/env python

from __future__ import print_function
import numpy as np


class OutputLayer:
    def __init__(self, instances, targets, lr=0.1, mt=0., wd=0., bs=0):
        # Parameters
        self.x = instances
        self.y = targets
        self.m = instances[0, :].size
        self.n = targets[0, :].size
        self.xsize = instances[:, 0].size
        self.c = np.zeros((1, self.n))  # i output
        self.w = np.zeros((self.n, self.m))
        self.lr = lr
        self.mt = mt
        self.wd = wd
        self.bs = bs
        # For Momentum
        self.delta_w = 0.
        self.delta_c = 0.

    def p_yx(self, x):
        wxb = np.dot(x, self.w.T) + self.c
        ewxb = np.exp(wxb)
        sumewxb = np.array([np.sum(ewxb, axis=1)]).T
        return ewxb / sumewxb

    def sim(self, x):
        in_size = x[:, 0].size
        p = self.p_yx(x)
        p = np.argmax(p, axis=1)
        p_fmt = np.zeros((in_size, self.n))
        for i in range(in_size):
            p_fmt[i, p[i]] = 1
        return p_fmt

    def train(self):
        batch_size = self.bs
        if self.bs < 1:
            return self.train_by_mini_batch(self.x, self.y, self.xsize)
        vint = np.vectorize(lambda q: int(q))
        indices = np.arange(0, self.xsize)
        indices = vint(indices / batch_size)
        num_batch = np.max(indices) + 1
        for i in range(num_batch):
            mini_batch = self.x[indices == i, :]
            mini_target = self.y[indices == i, :]
            mini_size = mini_batch[:, 0].size
            self.train_by_mini_batch(mini_batch, mini_target, mini_size)

    def train_by_mini_batch(self, x, y, xsize):
        pyx = self.p_yx(x)
        dy = y - pyx
        alpha = self.lr / xsize
        self.delta_w = alpha*np.dot(dy.T, x) + self.mt*self.delta_w - self.wd*self.w
        self.delta_c = alpha*np.sum(dy, axis=0)
        self.w += self.delta_w
        self.c += self.delta_c

    def negative_log_likelihood(self):
        pyx = self.p_yx(self.x)
        y = np.argmax(self.y, axis=1)
        l = (np.log(pyx))[range(self.xsize), y]
        return -np.mean(l)
