#!/usr/bin/env python

from __future__ import print_function
import NeuralLayer as Nl
import numpy as np
import numpy.random as rnd


class SigmoidLayer(Nl.NeuralLayer):
    def __init__(self, input_size, output_size,
                 learning_rate=0.1, momentum=0.5, weight_decay=0.001):
        Nl.NeuralLayer.__init__(self, input_size, output_size,
                                learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay)

    def initialize_params(self, x):
        p = np.mean(x, axis=0)
        v_log = np.vectorize(lambda q: 0 if q == 0 or q == 1 else np.log(q/(1-q)))
        self.b += v_log(p)
        self.w += rnd.normal(scale=0.01, size=(self.n, self.m))

    def train_unsupervised(self, x):
        v_sigmoid = np.vectorize(lambda q: 1 / (1 + np.exp(-q)))
        v_binarize = np.vectorize(lambda q, t: 1 if q < t else 0)
        batch_size = x.shape[0]
        w_T = self.w.T
        y0 = np.random.rand(batch_size, self.n)
        x1 = np.random.rand(batch_size, self.m)
        p_yx0 = v_sigmoid(np.dot(x, w_T) + self.c)
        y0 = v_binarize(y0, p_yx0)
        p_xy0 = v_sigmoid(np.dot(y0, self.w) + self.b)
        x1 = v_binarize(x1, p_xy0)
        p_yx1 = v_sigmoid(np.dot(x1, w_T) + self.c)
        dw = np.dot(p_yx0.T, x) - np.dot(p_yx1.T, x1)
        db = x - x1
        dc = p_yx0 - p_yx1
        db = np.sum(db, axis=0)
        dc = np.sum(dc, axis=0)
        alpha = self.lr / float(batch_size)
        self.delta_w = alpha*dw + self.mt*self.delta_w - self.wd*self.w
        self.delta_b = alpha*db + self.mt*self.delta_b - self.wd*self.b
        self.delta_c = alpha*dc + self.mt*self.delta_c - self.wd*self.c
        self.w += self.delta_w
        self.b += self.delta_b
        self.c += self.delta_c

    def train_with_delta(self, x, output_delta):
        batch_size = x.shape[0]
        dw = np.dot(output_delta.T, x)
        dc = np.sum(output_delta, axis=0)
        alpha = self.lr / float(batch_size)
        self.delta_w = alpha*dw + self.mt*self.delta_w - self.wd*self.w
        self.delta_c = alpha*dc + self.mt*self.delta_c - self.wd*self.c
        self.w += self.delta_w
        self.c += self.delta_c

    def generate(self, x):
        v_sigmoid = np.vectorize(lambda q: 1 / (1 + np.exp(-q)))
        p_yx0 = v_sigmoid(np.dot(x, self.w.T) + self.c)
        p_xy0 = v_sigmoid(np.dot(p_yx0, self.w) + self.b)
        return p_xy0

    def simulate(self, x):
        v_sigmoid = np.vectorize(lambda q: 1 / (1 + np.exp(-q)))
        p_yx0 = v_sigmoid(np.dot(x, self.w.T) + self.c)
        return p_yx0

    def predict(self, x):
        v_binarize = np.vectorize(lambda q, t: 1 if q < t else 0)
        return v_binarize(self.simulate(x))

    def get_input_delta(self, output_delta):
        input_delta = np.dot(output_delta, self.w)
        return input_delta

