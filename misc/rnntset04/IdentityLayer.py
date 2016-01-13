#!/usr/bin/env python

from __future__ import print_function
import DummyLayer as Dl


class IdentityLayer(Dl.DummyLayer):
    def __init__(self):
        Dl.DummyLayer.__init__(self)

    def simulate(self, x):
        return x

    def predict(self, x):
        return x

    def get_delta(self, y, dedy):
        return dedy

    def get_dedx(self, delta):
        return delta

