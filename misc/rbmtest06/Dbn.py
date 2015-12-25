#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import RbmVectorized as rl
import OutputLayer as ol


class Dbn:
    def __init__(self,
                 instances=np.array([[1]]), targets=np.array([[1]]),
                 hidden_sizes=np.array([1]),
                 lr=0.1, mt=0., wd=0., bs=0):
        self.D = instances
        self.y = targets
        self.attr_size = instances[0, :].size
        self.feat_size = targets[0, :].size
        self.rbm_num = hidden_sizes.size
        self.m = np.zeros(self.rbm_num)
        self.n = np.zeros(self.rbm_num)
        paramshape = np.zeros(self.rbm_num+1).shape
        self.lr = self.force_shape_and_fill(lr, paramshape)
        self.mt = self.force_shape_and_fill(mt, paramshape)
        self.wd = self.force_shape_and_fill(wd, paramshape)
        self.bs = self.force_shape_and_fill(bs, paramshape)
        self.rbm = []
        for hl in range(self.rbm_num):
            self.m[hl] = self.attr_size if hl == 0 else hidden_sizes[hl-1]
            self.n[hl] = hidden_sizes[hl]
        self.outlayer = None

    @staticmethod
    def force_shape_and_fill(arg, shape, dtype=float):
        if not hasattr(arg, '__iter__'):
            return np.full(shape, arg, dtype=dtype)
        arg = np.array(arg)
        if arg.shape != shape:
            return np.full(shape, np.min(arg), dtype=dtype)
        return arg

    def local_rbm_train(self, rbm, epoch, verbose=0):
        for ep in range(epoch):
            rbm.train()
            if verbose >= 1:
                print('\r ep={}'.format(ep), end='')

    def pre_train(self, epoch=100, verbose=0):
        epoch = self.force_shape_and_fill(epoch, self.m.shape, dtype=int)
        self.rbm = []
        instances = self.D
        for i in range(self.rbm_num):
            rbm = rl.RbmVectorized(
                D=instances, m=self.m[i], n=self.n[i],
                lr=self.lr[i], mt=self.mt[i], wd=self.wd[i], bs=self.bs[i]
            )
            self.rbm.append(rbm)
            self.local_rbm_train(rbm, epoch[i], verbose)
            instances = rbm.sim_hidden(instances)
        return instances

    def fine_tune(self, instances, epoch=100, verbose=0):
        self.outlayer = ol.OutputLayer(
            instances=instances, targets=self.y,
            lr=self.lr[self.rbm_num], mt=self.mt[self.rbm_num], wd=self.wd[self.rbm_num],
            bs=self.bs[self.rbm_num]
        )
        for ep in range(epoch):
            self.outlayer.train()
            if verbose >= 1:
                print('\r ep={}'.format(ep), end='')

    def sim(self, x2):
        instances = x2
        for i in range(self.rbm_num):
            instances = self.rbm[i].sim_hidden(instances)
        predicts = self.outlayer.sim(instances)
        return predicts


def main(learningrate=0.4, epoch=1000, momentum=0.7, weight_decay=0.001, bath_size=0, verbose=1):
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
    label = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1]])
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
    tslbl = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0]])

    dbn = Dbn(instances=data, targets=label,
              hidden_sizes=np.array([4, 4, 4]),
              lr=learningrate, mt=momentum, wd=weight_decay, bs=bath_size
              )

    if verbose >= 1:
        print('Pre Train:')
    h_out = dbn.pre_train(epoch=epoch, verbose=verbose)
    if verbose >= 1:
        print(h_out)
        print('Fine Tune:')
    dbn.fine_tune(h_out, epoch=epoch, verbose=verbose)

    p = dbn.sim(tsdt)
    p_fmt = np.argmax(p, axis=1)
    t_fmt = np.argmax(tslbl, axis=1)

    err_num = np.count_nonzero(t_fmt - p_fmt)
    total_num = p_fmt.size

    print('')
    print(p_fmt)
    print(t_fmt)
    print('Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))

    ann = ol.OutputLayer(data, label, lr=learningrate, mt=momentum, wd=weight_decay, bs=bath_size)
    for ep in range(epoch):
        ann.train()
    p = ann.sim(tsdt)
    p_fmt = np.argmax(p, axis=1)
    err_num = np.count_nonzero(t_fmt - p_fmt)
    total_num = p_fmt.size
    print('')
    print(p_fmt)
    print(t_fmt)
    print('Err rate={}% ({}/{})'.format(err_num*100.0/total_num, err_num, total_num))


if __name__ == "__main__":
    main()

