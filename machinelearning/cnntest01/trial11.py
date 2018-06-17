#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from numpy.lib.stride_tricks import as_strided
import mnistloader as mn
from PIL import Image


def main():
    loader = mn.MNISTloader2D('__data__')
    [x, y] = loader.load_train_data(msize=300)
    print(x.shape)
    print(y.shape)
    xi = loader.render(x, width=20)
    img = Image.fromarray(np.uint8(xi * 255))
    img.show()

    # Sizes
    d = x.shape[0]
    f = x.shape[1]
    g = 1
    r = x.shape[2]
    c = x.shape[3]
    wh = 3
    ww = 3
    r2 = r - wh + 1
    c2 = c - ww + 1

    # Filter
    w = np.ones((f, g, wh, ww)) / (wh * ww)

    # Convolve
    x2 = as_strided(x, shape=(d, f, r2, c2, wh, ww), strides=(x.strides + x.strides[-2:]))
    h = np.einsum('lqijrs,qkrs->lkij', x2, w)
    hi = loader.render(h, width=20)
    img2 = Image.fromarray(np.uint8(hi * 255))
    img2.show()


if __name__ == "__main__":
    main()

