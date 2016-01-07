#!/usr/bin/env python

from __future__ import print_function
import SigmoidLayer as Sl
import numpy as np


# This class is for the output layer supposing the error function is the cross-entropy
class SigmoidOutputLayer(Sl.SigmoidLayer):
    def __init__(self, output_size,
                 learning_rate=0.1, momentum=0.5, weight_decay=0.001):
        Sl.SigmoidLayer.__init__(self, output_size,
                                 learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)

    def get_input_delta(self, x, y, output_delta):
        input_delta = np.dot(output_delta, self.w)
        return input_delta

