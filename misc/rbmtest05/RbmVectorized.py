#!/usr/bin/env python

from __future__ import print_function
import Rbm as rl
import pyopencl as cl
import numpy as np


class RbmVectorized(rl.Rbm):
    def __init__(self, D=None, m=0, n=0, lr=0.1, mt=0., wd=0., bs=0):
        rl.Rbm.__init__(self, D, m, n, lr, mt, wd, bs)

    def cd_1_batch(self, v0, batch_size):
        hs = np.random.rand(batch_size, self.n)
        vs = np.random.rand(batch_size, self.m)
        hs = self.activate(hs, self.p_hv(v0))
        vs = self.activate(vs, self.p_vh(hs))

        return [hs, vs]

    def train_by_mini_batch(self, batch, batch_size):
        dw = np.zeros((self.n, self.m))
        db = np.zeros((1, self.m))
        dc = np.zeros((1, self.n))
        v0 = batch
        [_, vk] = self.cd_1_batch(v0, batch_size)  # Fix k = 1
        p_hv0 = self.p_hv(v0)
        p_hvk = self.p_hv(vk)
        dw = np.dot(p_hv0.T, v0) - np.dot(p_hvk.T, vk)
        db = v0 - vk
        dc = p_hv0 - p_hvk
        alpha = self.lr / batch_size
        db = np.sum(db, axis=0)
        dc = np.sum(dc, axis=0)
        self.delta_w = alpha*dw + self.mt*self.delta_w - self.wd*self.w
        self.delta_b = alpha*db + self.mt*self.delta_b - self.wd*self.b
        self.delta_c = alpha*dc + self.mt*self.delta_c - self.wd*self.c
        self.w += self.delta_w
        self.b += self.delta_b
        self.c += self.delta_c

