#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


def pca(x, dim_y):
    n, dim_x = x.shape
    assert dim_y < dim_x
    xm = x - np.mean(x, axis=0)
    xm_xmt = np.dot(xm, xm.transpose())
    e, v = np.linalg.eig(xm_xmt)
    # Pick the dim_y th largest ones
    idx = (np.argsort(e)[::-1])[:dim_y]
    e, v = e[idx], v[:, idx]
    e = np.ones(e.shape) / np.sqrt(e)
    a = np.diag(e)
    u = np.dot(np.dot(xm.transpose(), v), a)
    y = np.dot(xm, u)
    return y, u


def lda(x, dim_y):
    c = len(x)
    n = np.array([x[i].shape[0] for i in range(c)])
    dim_x = x[0].shape[1]
    sw = np.zeros((dim_x, dim_x))
    mu = np.zeros((dim_x,))
    for i in range(c):
        m = np.mean(x[i], axis=0)
        mu += m
        xm = x[i] - m
        sw += np.dot(xm.transpose(), xm) / float(n[i])
    mu /= float(c)
    sb = np.zeros((dim_x, dim_x))
    for i in range(c):
        mm = np.array([np.mean(x[i], axis=0) - mu])
        sb += np.dot(mm.transpose(), mm) * (n[i] - 1)
    sw1_sb = np.dot(np.linalg.inv(sb), sw)
    e, v = np.linalg.eig(sw1_sb)
    # Pick the dim_y th largest ones
    idx = (np.argsort(e)[::-1])[:dim_y]
    v = v[:, idx]
    return v


def test_pca():
    x = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [5, 5, 5]
    ])
    y, _ = pca(x, 1)
    print(y)
    plt.scatter(y, np.zeros(y.shape))
    plt.show()
    y, _ = pca(x, 2)
    print(y)
    plt.scatter(y[:, 0], y[:, 1])
    plt.show()

    theta = np.pi / float(4)
    n = 1000
    r2 = np.random.normal(loc=5, scale=1.0, size=(n,))
    x2 = r2 * np.cos(theta + 0.2*(np.random.random(size=(n,))))
    y2 = r2 * np.sin(theta + 0.2*(np.random.random(size=(n,))))
    x = []
    for i in range(n):
        x.append([x2[i], y2[i]])
    x = np.array(x)
    y, u = pca(x, 1)
    plt.subplot(2, 1, 1)
    plt.scatter(x2, y2)
    for i in range(u.shape[1]):
        mu = x.mean(axis=0)
        a = u[1, i] / u[0, i]
        b = mu[1] - (mu[0]*a)
        f = lambda q: (a*q + b)
        hg = np.arange(np.min(x2), np.max(x2), 0.1)
        plt.plot(hg, f(hg), color="r")
    plt.subplot(2, 1, 2)
    plt.scatter(y, np.zeros(y.shape))
    plt.show()


def test_lda():
    x, xp, yp, mp = [], [], [], []
    c = 5
    xmin, xmax = 100000, -100000
    for i in range(c):
        xm = np.random.random() * 20 - 10
        xs = np.random.random() * 2
        ym = np.random.random() * 20 - 10
        ys = np.random.random() * 2
        n = np.random.randint(500, 700)
        x2 = np.random.normal(xm, xs, n)
        y2 = np.random.normal(ym, ys, n)
        xtmp = []
        for j in range(n):
            xtmp.append([x2[j], y2[j]])
        x.append(np.array(xtmp))
        xp.append(x2)
        yp.append(y2)
        mp.append(np.mean(np.array(xtmp), axis=0))
        xmin = min(xmin, np.min(x2))
        xmax = max(xmax, np.max(x2))

    u = lda(x, 2)
    cstr = ["b", "g", "y", "c", "m"]
    for k in range(c):
        plt.scatter(xp[k], yp[k], color=cstr[k])
    mu = np.mean(np.array(mp), axis=0)
    for i in range(u.shape[1]):
        a = u[1, i] / u[0, i]
        b = mu[1] - (mu[0]*a)
        f = lambda q: (a*q + b)
        hg = np.arange(xmin, xmax, 0.1)
        plt.plot(hg, f(hg), color="r")
    plt.plot(mu[0], mu[1], 'o', color="r")
    plt.show()

for kk in range(10):
    test_lda()

