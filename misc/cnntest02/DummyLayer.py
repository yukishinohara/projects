#!/usr/bin/env python

from __future__ import print_function


class DummyLayer:
    def __init__(self):
        pass

    def initialize_params(self, x):
        pass

    def train_unsupervised(self, x):
        pass

    def train_with_delta(self, x, output_delta):
        pass

    def simulate(self, x):
        pass

    def predict(self, x):
        pass

    def get_input_delta(self, output_delta):
        pass
