#!/usr/bin/env python

from __future__ import print_function
import Rbm as rl
import numpy as np


class RbmVectorized(rl.Rbm):
    def __init__(self, D=None, m=0, n=0, lr=0.1, mt=0., wd=0., bs=0):
        rl.Rbm.__init__(self, D, m, n, lr, mt, wd, bs)

    def train_by_mini_batch(self, batch, batch_size):
        v_sigmoid = np.vectorize(lambda y: 1/(1+np.exp(-y)))
        v_activate = np.vectorize(lambda y, t: 1 if y < t else 0)
        wT = self.w.T
        h0 = np.random.rand(batch_size, self.n)
        v1 = np.random.rand(batch_size, self.m)
        p_hv0 = v_sigmoid(np.dot(batch, wT) + self.c)
        h0 = v_activate(h0, p_hv0)
        p_vh0 = v_sigmoid(np.dot(h0, self.w) + self.b)
        v1 = v_activate(v1, p_vh0)
        p_hv1 = v_sigmoid(np.dot(v1, wT) + self.c)
        dw = np.dot(p_hv0.T, batch) - np.dot(p_hv1.T, v1)
        db = batch - v1
        dc = p_hv0 - p_hv1
        db = np.sum(db, axis=0)
        dc = np.sum(dc, axis=0)
        alpha = self.lr / float(batch_size)
        self.delta_w = alpha*dw + self.mt*self.delta_w - self.wd*self.w
        self.delta_b = alpha*db + self.mt*self.delta_b - self.wd*self.b
        self.delta_c = alpha*dc + self.mt*self.delta_c - self.wd*self.c
        self.w += self.delta_w
        self.b += self.delta_b
        self.c += self.delta_c


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

    rbm = RbmVectorized(data, m=6, n=2, lr=learningrate, mt=momentum, wd=weight_decay)

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

