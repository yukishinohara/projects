#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import RbmVectorized as rl
import time
import mnistloader as mn


def main(verbose=2):
    loader = mn.MNISTloader('.\\__data__')
    [x, y] = loader.load_train_data(msize=3000)
    x2 = np.c_[y, x]
    m = x2[0, :].size
    f_size = y[0, :].size
    print(x.shape)
    print(y.shape)

    # Params
    hidden = 300
    ita = 0.07
    epoch = 250
    momentum = 0.6
    weight_decay = 0.001
    batch_size = 100

    rbm = rl.RbmVectorized(D=x2, m=m, n=hidden, lr=ita, mt=momentum, wd=weight_decay, bs=batch_size)
    start = time.time()
    for ep in range(epoch):
        rbm.train()
        if verbose >= 3 and ep % 10 == 1:
            print('\r ep={}, E={}'.format(ep, rbm.cost()))
        elif verbose >= 2:
            print('\r ep={}'.format(ep), end="")
    train_time = time.time() - start

    # Test
    [xt, yt] = loader.load_test_data(msize=1000)
    zt = np.zeros(yt.shape)
    xt2 = np.c_[zt, xt]
    p = rbm.sim2(xt2)
    p2 = p[:, 0:f_size]

    yf = np.argmax(yt, axis=1)
    pf = np.argmax(p2, axis=1)
    err = np.count_nonzero(pf - yf)
    err_rate = float(err) / yf.size
    print('')
    print(yf)
    print(pf)

    print('Vector: time={}, er={}% ({}/{})'.format(train_time, err_rate*100, err, yf.size))


if __name__ == "__main__":
    main()

