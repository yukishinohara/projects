#!/usr/bin/env python

import random as rnd
import numpy as np


class Rbm:
  def __init__(self, D=None, m=0, n=0, a=0.1):
    self.D = D
    self.m = m
    self.n = n
    self.b = np.zeros(m) #j visible
    self.c = np.zeros(n) #i hidden
    self.w = np.zeros((n, m))
    self.a = a

  def sigmoid(self, x):
    return 1. / (1. + np.exp(-x))

  def p_vh(self, h, k):
    return self.sigmoid(
        np.dot(self.w[:,k], h) + self.b[k])

  def p_hv(self, v, k):
    return self.sigmoid(
        np.dot(self.w[k,:], v) + self.c[k])

  def cd_k(self, v0):
    h1 = np.zeros(self.n)
    v1 = np.zeros(self.m)

    for i in range(self.n):
      h1[i] = 0 if rnd.random() > self.p_hv(v0, i) else 1

    for j in range(self.m):
      v1[j] = 0 if rnd.random() > self.p_vh(h1, j) else 1

    return [h1, v1]

  def train(self):
    dsize = np.size(self.D[:,1])
    dw = np.zeros((self.n, self.m))
    db = np.zeros(self.m)
    dc = np.zeros(self.n)
    for l in range(dsize):
      v0 = self.D[l,:]
      [hk, vk] = self.cd_k(v0)  # k = 1
      for j in range(self.m):
        db[j] = db[j] + v0[j] - vk[j]
        for i in range(self.n):
          dc[i] = dc[i] + self.p_hv(v0, i) - self.p_hv(vk, i)
          dw[i,j] = dw[i,j] + (self.p_hv(v0, i) * v0[j]) - (self.p_hv(vk, i) * vk[j])

    alpha = self.a / dsize
    self.w = alpha*self.w + dw
    self.b = alpha*self.b + db
    self.c = alpha*self.c + dc

  def test(self, v):
    h1 = np.zeros(self.n)
    v1 = np.zeros(self.m)
    for i in range(self.n):
      h1[i] = self.sigmoid(np.dot(self.w[i,:], v) + self.c[i])

    for j in range(self.m):
      v1[j] = self.sigmoid(np.dot(self.w[:,j], h1) + self.b[j])

    dispstr='E={}'.format(self.energy(v1,h1))
    print dispstr

    return v1

  def energy(self, v, h):
    t1 = 0
    t2 = 0
    t3 = 0
    for i in range(self.n):
      t3 = t3 + self.c[i]*h[i]
      for j in range(self.m):
        t1 = t1 + self.w[i,j]*h[i]*v[j]

    for j in range(self.m):
      t2 = t2 + self.b[j]*v[j]

    return -t1-t2-t3
  

def main(learningrate=0.01,epoch=250):
    data = np.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,1,0]])

    rbm = Rbm(data, m=6, n=5, a=learningrate)

    for ep in range(epoch):
      rbm.train()
      dispstr = '\rEp:{}'.format(ep)
      print dispstr ,

    rbm.train()
    print rbm.test([0, 0, 1, 1, 1, 0])
    print rbm.test([0, 1, 1, 0, 0, 0])
    print rbm.test([0, 0, 0, 0, 0, 0])
    print rbm.test([0, 0, 1, 0, 0, 1])

    dispstr = 'w={} b={} c={}'.format(rbm.w, rbm.b, rbm.c)
    print dispstr

if __name__ == "__main__":
    main()

