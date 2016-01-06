#!/usr/bin/env python

from __future__ import print_function
import os
import struct
import numpy as np


class MNISTloader2D:
    def __init__(self, path='.'):
        self.tran_img_fname = os.path.join(path, 'train-images.idx3-ubyte')
        self.tran_lbl_fname = os.path.join(path, 'train-labels.idx1-ubyte')
        self.test_img_fname = os.path.join(path, 't10k-images.idx3-ubyte')
        self.test_lbl_fname = os.path.join(path, 't10k-labels.idx1-ubyte')

    def load_test_data(self, msize=0, th=50):
        return self.load(self.test_img_fname, self.test_lbl_fname, msize, th)

    def load_train_data(self, msize=0, th=50):
        return self.load(self.tran_img_fname, self.tran_lbl_fname, msize, th)

    def load(self, img_fname, lbl_fname, msize, th):
        img_file = open(img_fname, 'rb')
        lbl_file = open(lbl_fname, 'rb')
        fm_size = 1

        [magic, d, rows, cols] = struct.unpack('>IIII', img_file.read(16))
        assert magic == 2051
        d = d if msize == 0 or d < msize else msize
        img_data = np.zeros((d, fm_size, rows, cols))
        for l in range(d):
            for k in range(fm_size):
                for i in range(rows):
                    img_data[l, k, i, :] = np.array(
                            struct.unpack('>{}B'.format(cols), img_file.read(cols)),
                            dtype=int
                    )
        img_file.close()

        [magic, l_size] = struct.unpack('>II', lbl_file.read(8))
        assert magic == 2049
        assert l_size >= d
        lbl_data = np.array(struct.unpack('>{}B'.format(d), lbl_file.read(d)), dtype=int)
        lbl_file.close()

        vzerone = np.vectorize(lambda q: 1 if q > th else 0)
        img_data = vzerone(img_data)

        lbl_data_fmt = np.zeros((d, np.max(lbl_data)+1))
        for i in range(d):
            lbl_data_fmt[i, lbl_data[i]] = 1

        return img_data, lbl_data_fmt

    def render_console(self, img):
        img_file = open(self.tran_img_fname, 'rb')
        [magic, _, rows, cols] = struct.unpack('>IIII', img_file.read(16))
        assert magic == 2051
        img_file.close()
        num = img[:, 0].size
        for k in range(num):
            for i in range(rows):
                for j in range(cols):
                    pixel = '...' if img[k, i*rows + j] == 0 else '###'
                    print(pixel, end='')
                print('')
            print('')

    def render(self, img_data, msize=None, width=10, f=0):
        if msize is None:
            msize = img_data.shape[0]
        d = min(img_data.shape[0], msize)
        r = img_data.shape[2]
        c = img_data.shape[3]
        width = min(width, d)
        height = np.ceil(float(d) / float(width))
        ans = np.zeros((height * r, width * c))
        for l in range(d):
            for i in range(r):
                for j in range(c):
                    sj = (l % width) * r
                    si = (np.floor(l / width)) * c
                    ans[si + i, sj + j] = img_data[l, f, i, j]
        return ans
