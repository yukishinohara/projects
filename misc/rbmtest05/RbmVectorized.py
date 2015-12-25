#!/usr/bin/env python

from __future__ import print_function
import Rbm as rl
import numpy as np


class RbmVectorized(rl.Rbm):
    def __init__(self, D=None, m=0, n=0, lr=0.1, mt=0., wd=0., bs=0):
        rl.Rbm.__init__(self, D, m, n, lr, mt, wd, bs)
        self.v_sigmoid = np.vectorize(lambda y: 1/(1+np.exp(-y)))
        self.v_activate = np.vectorize(lambda y, t: 1 if y < t else 0)

    def train_by_mini_batch(self, batch, batch_size):
        wT = self.w.T
        h0 = np.random.rand(batch_size, self.n)
        v1 = np.random.rand(batch_size, self.m)
        p_hv0 = self.v_sigmoid(np.dot(batch, wT) + self.c)
        h0 = self.v_activate(h0, p_hv0)
        p_vh0 = self.v_sigmoid(np.dot(h0, self.w) + self.b)
        v1 = self.v_activate(v1, p_vh0)
        p_hv1 = self.v_sigmoid(np.dot(v1, wT) + self.c)
        dw = np.dot(p_hv0.T, batch) - np.dot(p_hv1.T, v1)
        db = batch - v1
        dc = p_hv0 - p_hv1
        db = np.sum(db, axis=0)
        dc = np.sum(dc, axis=0)
        alpha = self.lr / float(batch_size)
        self.delta_w = alpha*dw + self.mt*self.delta_w - self.wd*self.w
        self.delta_b = alpha*db + self.mt*self.delta_b - self.wd*self.b
        self.delta_c = alpha*dc + self.mt*self.delta_c - self.wd*self.c
        self.w += self.delta_w
        self.b += self.delta_b
        self.c += self.delta_c

