#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import DummyLayer as Dl


class NeuralLayer(Dl.DummyLayer):
    def __init__(self, input_size, output_size,
                 learning_rate=0.1, momentum=0.5, weight_decay=0.001):
        # Parameters
        self.m = input_size
        self.n = output_size
        self.lr = learning_rate
        self.mt = momentum
        self.wd = weight_decay
        self.w = np.zeros((self.n, self.m))
        self.b = np.zeros((1, self.m))
        self.c = np.zeros((1, self.n))
        # For the momentum term
        self.delta_w = 0.
        self.delta_b = 0.
        self.delta_c = 0.
        Dl.DummyLayer.__init__(self)

    def initialize_params(self, x):
        self.w += 0.
        self.b += 0.
        self.c += 0.
        return

    def set_hyper_params(self, learning_rate=None, momentum=None, weight_decay=None):
        if learning_rate is not None:
            self.lr = learning_rate
        if momentum is not None:
            self.mt = momentum
        if weight_decay is not None:
            self.wd = weight_decay

    def train_unsupervised(self, x):
        pass

    def train_with_delta(self, x, output_delta):
        pass

    def simulate(self, x):
        return np.zeros((1, self.n))

    def predict(self, x):
        return np.zeros((1, self.n))

    def get_input_delta(self, output_delta):
        return np.zeros((1, self.m))

