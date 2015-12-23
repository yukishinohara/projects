#!/usr/bin/env python

import numpy.random as rnd
import numpy as np


class Rbm:
    def __init__(self, D=None, m=0, n=0, a=0.1, u=0.):
        # Parameters
        self.D = D
        self.m = m
        self.n = n
        self.b = np.zeros((1, m))  # j visible
        self.c = np.zeros((1, n))  # i hidden
        self.w = np.zeros((n, m))
        self.a = a
        self.u = u
        # For Momentum
        self.delta_w = 0.
        self.delta_b = 0.
        self.delta_c = 0.

    @staticmethod
    def sigmoid(x):
        vf = np.vectorize(lambda y: 1/(1+np.exp(-y)))
        return vf(x)

    @staticmethod
    def sgn(x, th):
        vf = np.vectorize(lambda y, t: 1 if y < t else 0)
        return vf(x, th)

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
            hs = self.sgn(hs, self.p_hv(vb))
            vs = self.sgn(vs, self.p_vh(hs))
            vb = vs.copy()

        return [hs, vs]

    def train(self):
        dsize = np.size(self.D[:, 1])
        dw = np.zeros((self.n, self.m))
        db = np.zeros((1, self.m))
        dc = np.zeros((1, self.n))
        for l in range(dsize):
            v0 = self.D[l:l+1, :]
            [_, vk] = self.cd_k(v0)  # k = 1
            p_hv0 = self.p_hv(v0)
            p_hvk = self.p_hv(vk)
            dw += np.dot(p_hv0.T, v0) - np.dot(p_hvk.T, vk)
            db += v0 - vk
            dc += p_hv0 - p_hvk
        alpha = self.a / dsize
        self.delta_w = alpha*dw + self.u*self.delta_w
        self.delta_b = alpha*db + self.u*self.delta_b
        self.delta_c = alpha*dc + self.u*self.delta_c
        self.w += self.delta_w
        self.b += self.delta_b
        self.c += self.delta_c

    def sim(self, v):
        v1 = self.sim2(v)
        return self.sgn(np.full(v1.shape, 0.5), v1)

    def sim2(self, v):
        h1 = self.p_hv(v)
        v1 = self.p_vh(h1)
        return v1

    def cost(self, v=None, lazy=False):
        if lazy:
            return self.log_pl(v)
        return self.log_pl_precise(v)

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


def main(learningrate=0.6, epoch=300, verbose=2):
    data = np.array([[1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 0]])

    tsdt = np.array([[0, 0, 1, 1, 1, 0],
                     [0, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1]])

    rbm = Rbm(data, m=6, n=2, a=learningrate)

    for ep in range(epoch):
        rbm.train()
        if verbose >= 2:
            print ('ep={}, E={}'.format(ep, rbm.log_pl_all()))

    if verbose >= 1:
        dsize = np.size(tsdt[:, 1])
        for l in range(dsize):
            [h1, v1] = rbm.cd_k(tsdt[l:l+1, :])
            energy = rbm.log_pl_precise(tsdt[l:l+1, :])
            print ('h={}, v={}, F={}'.format(h1, v1, energy))

    print (rbm.sim(tsdt))


if __name__ == "__main__":
    main()

