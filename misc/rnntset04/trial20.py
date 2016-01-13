#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import os
import Lstm
import NeuralNetwork as Nn
import shelve
import zipfile
import h5py as h5


def read_from_file(filename):
    fp = h5.File(filename, 'r')
    indices = fp['/train/indices'].value
    words = fp['/train/words'].keys()
    word_indices = {}
    indices_word = {}
    for index_str in words:
        index = int(index_str)
        word = fp['/train/words/{}'.format(index_str)].value
        if isinstance(word, np.ndarray):
            word = word[0]
        word_indices[word] = index
        indices_word[index] = word
    fp.close()
    return indices, word_indices, indices_word


def save_model(rnn):
    directory = os.path.join('.', '__trained__')
    if not os.path.exists(directory):
        os.makedirs(directory)
    datafile = shelve.open(os.path.join(directory, 'lstm_book'))
    datafile['lstm'] = rnn
    datafile.close()
    zf = zipfile.ZipFile('lstm_book.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(os.path.join('.', '__trained__')):
        for file in files:
            zf.write(os.path.join(root, file))
    zf.close()


def word2vector(word, size, word2index):
    ans = np.zeros((size,))
    ans[word2index[word]] = 1
    return ans


def main(verbose=1):
    # Read a processed text file
    x, word2index, index2word = read_from_file(os.path.join('.', 'testdata05.h5'))
    y = np.array(x[1:], dtype=np.int)
    x = np.array(x[:-1], dtype=np.int)

    # Common hyper parameters
    hid_lr = 0.08
    hid_mt = 0.7
    hid_wd = 0.007
    igt_lr = 0.07
    igt_mt = 0.6
    igt_wd = 0.009
    ogt_lr = 0.07
    ogt_mt = 0.6
    ogt_wd = 0.009
    fgt_lr = 0.07
    fgt_mt = 0.6
    fgt_wd = 0.009
    ep = 2
    repeat = 10000
    window = 0
    stride = 0

    # Create model
    nh = y.shape[1]
    ny = y.shape[1]
    hid_type = Nn.LAYER_TYPE_TANH
    hid_size = nh
    hid_param = {
        'output_size': hid_size,
        'learning_rate': hid_lr, 'momentum': hid_mt, 'weight_decay': hid_wd
    }
    igt_type = Nn.LAYER_TYPE_SIGMOID
    igt_size = nh
    igt_param = {
        'output_size': igt_size,
        'learning_rate': igt_lr, 'momentum': igt_mt, 'weight_decay': igt_wd
    }
    ogt_type = Nn.LAYER_TYPE_SIGMOID
    ogt_size = nh
    ogt_param = {
        'output_size': ogt_size,
        'learning_rate': ogt_lr, 'momentum': ogt_mt, 'weight_decay': ogt_wd
    }
    fgt_type = Nn.LAYER_TYPE_SIGMOID
    fgt_size = nh
    fgt_param = {
        'output_size': fgt_size,
        'learning_rate': fgt_lr, 'momentum': fgt_mt, 'weight_decay': fgt_wd
    }
    out_type = Nn.LAYER_TYPE_IDENTITY
    out_param = {
    }
    lstm = Lstm.Lstm(
        hid_type, hid_param, hid_size,
        out_type, out_param,
        igt_type, igt_param, igt_size,
        ogt_type, ogt_param, ogt_size,
        fgt_type, fgt_param, fgt_size
    )

    # Train generating some sentences
    period_idx = word2index['.']
    end_idx = word2index['END_SENTENCE']
    for rep in range(repeat):
        if verbose >= 1:
            print('Repeat: {} / {}'.format(rep, repeat))
        lstm.train(x, y, ep, window, stride, verbose)
        feat_size = y.shape[1]
        print('')
        current = 'BEGIN_SENTENCE'
        sentence_str = ''
        sentence = np.array([word2vector(current, feat_size, word2index)], dtype=np.int)
        for i in range(20):
            current = lstm.simulate(sentence)
            current[i, period_idx] /= 2
            current[i, end_idx] /= 3
            current = np.argmax(current[i])
            current = index2word[current]
            sentence = np.append(sentence, np.array([
                word2vector(current, feat_size, word2index)
            ]), axis=0)
            sentence_str = '{} {}'.format(sentence_str, current)
        print('{} '.format(sentence_str))
        current = index2word[np.random.randint(0, feat_size)]
        sentence_str = '' + current
        sentence = np.array([word2vector(current, feat_size, word2index)], dtype=np.int)
        for i in range(20):
            current = lstm.simulate(sentence)
            current[i, period_idx] /= 2
            current[i, end_idx] /= 3
            current = np.argmax(current[i])
            current = index2word[current]
            sentence = np.append(sentence, np.array([
                word2vector(current, feat_size, word2index)
            ]), axis=0)
            sentence_str = '{} {}'.format(sentence_str, current)
        print('{} '.format(sentence_str))
        if rep % 10 == 0:
            save_model(lstm)


if __name__ == "__main__":
    main()

