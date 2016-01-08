#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import DummyLayer as Dl


class ConvoPool2SigmoidLayer(Dl.DummyLayer):
    def __init__(self):
        # Parameters
        self.n = 1
        self.f = self.xh = self.xw = 1
        Dl.DummyLayer.__init__(self)

    def initialize_params(self, x, hyper_params):
        _, self.f, self.xh, self.xw = x.shape
        self.n = self.f * self.xh * self.xw

    def simulate(self, x):
        d, _, _, _ = x.shape
        y = np.reshape(x, (d, self.n))
        return y

    def predict(self, x):
        return self.simulate(x)

    def get_delta(self, y, dedy):
        return dedy

    def get_dedx(self, delta):
        d, _ = delta.shape
        dedx = np.reshape(delta, (d, self.f, self.xh, self.xw))
        return dedx

