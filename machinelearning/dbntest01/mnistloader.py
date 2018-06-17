#!/usr/bin/env python

from __future__ import print_function
import os
import struct
import numpy as np


class MNISTloader:
    def __init__(self, path='.'):
        self.tran_img_fname = os.path.join(path, 'train-images.idx3-ubyte')
        self.tran_lbl_fname = os.path.join(path, 'train-labels.idx1-ubyte')
        self.test_img_fname = os.path.join(path, 't10k-images.idx3-ubyte')
        self.test_lbl_fname = os.path.join(path, 't10k-labels.idx1-ubyte')

    def load_test_data(self, msize=0):
        return self.load(self.test_img_fname, self.test_lbl_fname, msize)

    def load_train_data(self, msize=0):
        return self.load(self.tran_img_fname, self.tran_lbl_fname, msize)

    def load(self, img_fname, lbl_fname, msize):
        img_file = open(img_fname, 'rb')
        lbl_file = open(lbl_fname, 'rb')

        [magic, isize, rows, cols] = struct.unpack('>IIII', img_file.read(16))
        assert magic == 2051
        isize = isize if msize == 0 or isize < msize else msize
        img_data = np.zeros((isize, (rows*cols)))
        for i in range(isize):
            img_data[i, :] = np.array(
                    struct.unpack('>{}B'.format(rows*cols), img_file.read(rows*cols)),
                    dtype=int
            )
        img_file.close()

        [magic, lsize] = struct.unpack('>II', lbl_file.read(8))
        assert magic == 2049
        assert lsize >= isize
        lsize = isize
        lbl_data = np.array(struct.unpack('>{}B'.format(lsize), lbl_file.read(lsize)), dtype=int)
        lbl_file.close()

        # vzerone = np.vectorize(lambda q: 1 if q != 0 else 0)
        img_data *= 0.00390625  # 1/256

        lbl_data_fmt = np.zeros((lsize, np.max(lbl_data)+1))
        for i in range(lsize):
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

