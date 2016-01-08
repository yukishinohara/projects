#!/usr/bin/env python

from __future__ import print_function


class DummyLayer:
    def __init__(self):
        pass

    def initialize_params(self, x, hyper_params):
        pass

    def update_hyper_params(self, hyper_params):
        pass

    def train_unsupervised(self, x):
        pass

    def train_with_delta(self, x, delta):
        pass

    def simulate(self, x):
        pass

    def predict(self, x):
        pass

    def get_delta(self, y, dedy):
        pass

    def get_dedx(self, delta):
        pass

