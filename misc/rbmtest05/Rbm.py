#!/usr/bin/env python

from __future__ import print_function
import numpy.random as rnd
import numpy as np


class Rbm:
    def __init__(self, D=None, m=0, n=0, lr=0.1, mt=0., wd=0., bs=0):
        # Parameters
        self.D = D
        self.m = m
        self.n = n
        self.b = np.zeros((1, m))  # j visible
        self.c = np.zeros((1, n))  # i hidden
        self.w = np.zeros((n, m))
        self.lr = lr
        self.mt = mt
        self.wd = wd
        self.bs = bs
        # For Momentum
        self.delta_w = 0.
        self.delta_b = 0.
        self.delta_c = 0.
        # Initial values for params
        self.initialize()

    @staticmethod
    def sigmoid(x):
        vf = np.vectorize(lambda y: 1/(1+np.exp(-y)))
        return vf(x)

    @staticmethod
    def activate(x, th):
        vf = np.vectorize(lambda y, t: 1 if y < t else 0)
        return vf(x, th)

    def initialize(self):
        p = np.mean(self.D, axis=0)
        self.b += np.log(p / (1-p))
        self.w += rnd.normal(scale=0.01, size=(self.n, self.m))

    def binarize(self, v):
        return self.activate(np.full(v.shape, 0.5), v)

    def p_vh(self, h):
        return self.sigmoid(np.dot(h, self.w) + self.b)

    def p_hv(self, v):
        return self.sigmoid(np.dot(v, self.w.T) + self.c)

    def cd_k(self, v0, k=1):
        k = 1 if k < 1 else k
        vb = v0.copy()
        hs = np.zeros((1, self.n))
        vs = np.zeros((1, self.m))
        for step in range(k):
            hs = rnd.rand(1, self.n)
            vs = rnd.rand(1, self.m)
            hs = self.activate(hs, self.p_hv(vb))
            vs = self.activate(vs, self.p_vh(hs))
            vb = vs.copy()

        return [hs, vs]

    def train(self):
        data_size = np.size(self.D[:, 1])
        batch_size = self.bs
        if self.bs < 1:
            return self.train_by_mini_batch(self.D, data_size)
        vint = np.vectorize(lambda q: int(q))
        indices = np.arange(0, data_size)
        indices = vint(indices / batch_size)
        num_batch = np.max(indices) + 1
        for i in range(num_batch):
            mini_batch = self.D[indices == i, :]
            mini_size = mini_batch[:, 0].size
            self.train_by_mini_batch(mini_batch, mini_size)

    def train_by_mini_batch(self, batch, batch_size):
        dw = np.zeros((self.n, self.m))
        db = np.zeros((1, self.m))
        dc = np.zeros((1, self.n))
        for l in range(batch_size):
            v0 = batch[l:l+1, :]
            [_, vk] = self.cd_k(v0)  # k = 1
            p_hv0 = self.p_hv(v0)
            p_hvk = self.p_hv(vk)
            dw += np.dot(p_hv0.T, v0) - np.dot(p_hvk.T, vk)
            db += v0 - vk
            dc += p_hv0 - p_hvk
        alpha = self.lr / batch_size
        self.delta_w = alpha*dw + self.mt*self.delta_w - self.wd*self.w
        self.delta_b = alpha*db + self.mt*self.delta_b - self.wd*self.b
        self.delta_c = alpha*dc + self.mt*self.delta_c - self.wd*self.c
        self.w += self.delta_w
        self.b += self.delta_b
        self.c += self.delta_c

    def sim(self, v):
        v1 = self.sim2(v)
        return self.binarize(v1)

    def sim2(self, v):
        h1 = self.p_hv(v)
        v1 = self.p_vh(h1)
        return v1

    def cost(self, v=None, lazy=False):
        if lazy:
            return self.log_pl(v)
        return self.log_pl_precise(v)

    def test_v(self, v):
        h1 = self.p_hv(v)
        h2 = self.binarize(h1)
        v1 = self.p_vh(h1)
        v2 = self.binarize(v1)
        cost = self.cost(v)
        return [v1, h1, v2, h2, cost]

    def free_energy(self, v):
        t1 = np.dot(self.b, v.T)
        h_1 = np.exp(np.dot(v, self.w.T) + self.c)
        h_0 = np.full(h_1.shape, 1.)
        t2 = np.sum(np.log(h_0 + h_1), axis=1)
        return -t1-t2

    def log_pl_for(self, i, D=None):
        D = self.D if D is None else D
        Di = D.copy()
        Di[:, i] = 1 - Di[:, i]
        fe_D  = self.free_energy(D)
        fe_Di = self.free_energy(Di)
        costs = self.m * np.log(self.sigmoid(fe_Di - fe_D))
        return np.mean(costs)

    def log_pl_all(self, D=None):
        idx = np.arange(self.m)
        v_log_pl = np.vectorize(self.log_pl_for, excluded=[1])
        return v_log_pl(idx, D)

    def log_pl(self, D=None):
        i = rnd.randint(self.m)
        return self.log_pl_for(i, D)

    def log_pl_precise(self, D=None):
        return np.mean(self.log_pl_all(D))


def main(learningrate=0.4, epoch=1000, momentum=0.7, weight_decay=0.001, verbose=2):
    # Attempt to learn 3 types of data [1,1,0,0,0,0], [0,0,1,1,0,0], [0,0,0,0,1,1] with some noise.
    # It sometimes fails to learn them converging to another local minimum.
    data = np.array([[1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1],
                     [1, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 1, 1]])

    tsdt = np.array([[1, 1, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1],
                     [1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 0, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0]])

    rbm = Rbm(data, m=6, n=2, lr=learningrate, mt=momentum, wd=weight_decay)

    for ep in range(epoch):
        rbm.train()
        if verbose >= 2:
            print('\rep={}, E={}'.format(ep, rbm.cost()), end="")
        elif verbose >= 1:
            print('\rep={}'.format(ep), end="")

    if verbose >= 1:
        dsize = np.size(tsdt[:, 1])
        print('')
        for l in range(dsize):
            v = tsdt[l:l+1, :]
            [v1, h1, v2, h2, cost] = rbm.test_v(v)
            print('{} test'.format(v))
            print('  h={}, v={}, F={}\n  h={}, v={}'.format(h2, v2, cost, h1, v1))

    print(rbm.sim(tsdt))


if __name__ == "__main__":
    main()

