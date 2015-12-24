#!/usr/bin/env python

from __future__ import print_function
import Rbm as rl
import pyopencl as cl


class RbmOpenCl(rl.Rbm):
    def __init__(self, D=None, m=0, n=0, lr=0.1, mt=0., wd=0., bs=0, device=None):
        rl.Rbm.__init__(self, D, m, n, lr, mt, wd, bs)
        self.ctx = \
            cl.create_some_context(interactive=True) if device is None else cl.Context(device)

    def train_by_mini_batch(self, batch, batch_size):
        x = self.D

