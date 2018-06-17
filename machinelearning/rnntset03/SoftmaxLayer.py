#!/usr/bin/env python

from __future__ import print_function
import IdentityLayer as Nl
import numpy as np


class SoftmaxLayer(Nl.IdentityLayer):
    def __init__(self):
        self.n = 0
        Nl.IdentityLayer.__init__(self)

    def initialize_params(self, x, hyper_params):
        Nl.IdentityLayer.initialize_params(self, x, hyper_params)
        self.n = x.shape[1]

    def simulate(self, x):
        a = np.exp(x)
        p_yx0 = a / (np.sum(a, axis=1).reshape((a.shape[0], 1)))
        return p_yx0

    def predict(self, x):
        return self.simulate(x)

    def get_delta(self, y, dedy):
        return dedy

    def get_dedx(self, delta):
        return delta

