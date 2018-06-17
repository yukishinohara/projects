#!/usr/bin/env python

from __future__ import print_function
import NeuralLayer as Nl
import numpy as np
import numpy.random as rnd


class HyperbolicTangentLayer(Nl.NeuralLayer):
    def __init__(self, output_size,
                 learning_rate=0.1, momentum=0.5, weight_decay=0.001):
        Nl.NeuralLayer.__init__(self, output_size,
                                learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)

    def initialize_params(self, x, hyper_params):
        Nl.NeuralLayer.initialize_params(self, x, hyper_params)
        bound = self.m + self.n
        bound = 1. / np.sqrt(bound)
        self.w += rnd.uniform(-bound, bound, size=(self.n, self.m))

    def train_with_delta(self, x, delta):
        batch_size = x.shape[0]
        dw = np.dot(delta.T, x)
        dc = np.sum(delta, axis=0)
        alpha = self.lr / float(batch_size)
        self.delta_w = alpha*dw + self.mt*self.delta_w - self.wd*self.w
        self.delta_c = alpha*dc + self.mt*self.delta_c - self.wd*self.c
        self.w += self.delta_w
        self.c += self.delta_c

    def generate(self, x):
        p_yx0 = np.tanh(np.dot(x, self.w.T) + self.c)
        p_xy0 = np.tanh(np.dot(p_yx0, self.w) + self.b)
        return p_xy0

    def simulate(self, x):
        p_yx0 = np.tanh(np.dot(x, self.w.T) + self.c)
        return p_yx0

    def predict(self, x):
        v_binarize = np.vectorize(lambda q, t: 1 if q < t else 0)
        return v_binarize(self.simulate(x))

    def get_delta(self, y, dedy):
        ones = np.ones(y.shape)
        delta = dedy * (ones - (y*y))
        return delta

    def get_dedx(self, delta):
        return np.dot(delta, self.w)

