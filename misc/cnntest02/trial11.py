#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from numpy.lib.stride_tricks import as_strided
import mnistloader as mn
from PIL import Image


def main():
    loader = mn.MNISTloader2D('__data__')
    [x, y] = loader.load_train_data(msize=1000)
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

    # Back
    append_row = np.zeros((d, g, wh - 1, c2))
    append_col = np.zeros((d, g, r + wh - 1, ww - 1))
    h2 = np.append(
        np.append(append_row, h, axis=2),
        append_row, axis=2)
    h2 = np.append(
        np.append(append_col, h2, axis=3),
        append_col, axis=3)
    w2 = w[:, :, ::-1, ::-1]
    h3 = as_strided(h2, shape=(d, g, r, c, wh, ww), strides=(h2.strides + h2.strides[-2:]))
    rx = np.einsum('lpijrs,kprs->lkij', h3, w2)
    rx = np.floor(rx * 255)
    rxi = loader.render(rx, width=20)
    img3 = Image.fromarray(np.uint8(rxi))
    img3.show()


if __name__ == "__main__":
    main()

