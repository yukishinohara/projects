#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import os
import Rnn as Rn
import NeuralNetwork as Nn
import nltk
import shelve
import zipfile
import h5py as h5


def read_text(num_vocab=5000, filename='alice_in_wonderland.txt', outfilename='testdata04.h5'):
    token_begin = 'BEGIN_SENTENCE'
    token_end = 'END_SENTENCE'
    token_unknown = 'UNKNOWN_WORD'
    filename = os.path.join('.', '__data__', filename)
    fp = open(filename)
    all_str = fp.read()
    fp.close()
    sentences = nltk.tokenize.sent_tokenize(all_str.lower())
    sentences = ['{} {} {}'.format(token_begin, s, token_end) for s in sentences]
    words = []
    for s in sentences:
        words.extend(nltk.tokenize.word_tokenize(s))
    freq = nltk.FreqDist(words)
    print('Read {}: contains {} unique words'.format(filename, len(freq.items())))
    vocab = freq.most_common(num_vocab)
    word_indices = {token_unknown: 0}
    indices_word = {0: token_unknown}
    for i, w in enumerate(vocab):
        word_indices[w[0]] = i + 1
        indices_word[i + 1] = w[0]
    indices = np.zeros((len(words), len(vocab) + 1), dtype=np.int)
    for i, w in enumerate(words):
        if w in word_indices:
            indices[i, word_indices[w]] = 1
        else:
            indices[i, word_indices[token_unknown]] = 1

    output_file = os.path.join('.', outfilename)
    fp = h5.File(output_file, 'w')
    train_h5 = fp.create_group('train')
    train_h5.create_dataset('indices',
                            shape=indices.shape,
                            dtype=indices.dtype,
                            data=indices,
                            compression="gzip")
    word_indices_h5 = train_h5.create_group('words')
    for kw in word_indices:
        val = word_indices[kw]
        word_indices_h5.create_dataset('{}'.format(val),
                                       shape=(1,),
                                       data=kw,
                                       compression="gzip")
    fp.close()

    return indices, word_indices, indices_word


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
    datafile = shelve.open(os.path.join(directory, 'rnn_book'))
    datafile['rnn'] = rnn
    datafile.close()
    zf = zipfile.ZipFile('rnn_book.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(os.path.join('.', '__trained__')):
        for file in files:
            zf.write(os.path.join(root, file))
    zf.close()


def word2vector(word, size, word2index):
    ans = np.zeros((size,))
    ans[word2index[word]] = 1
    return ans


def main(verbose=1):
    # x, word2index, index2word = read_text(filename='tiny_esl3.txt', outfilename='testdata06.h5')
    x, word2index, index2word = read_from_file(os.path.join('.', 'testdata05.h5'))
    y = np.array(x[1:], dtype=np.int)
    x = np.array(x[:-1], dtype=np.int)

    hid_lr = 0.5
    hid_mt = 0.5
    hid_wd = 0.0001
    ep = 3
    repeat = 10000
    window = 0
    stride = 0

    hid_type = Nn.LAYER_TYPE_TANH
    hid_size = y.shape[1]
    hid_param = {
        'output_size': hid_size,
        'learning_rate': hid_lr, 'momentum': hid_mt, 'weight_decay': hid_wd
    }
    out_type = Nn.LAYER_TYPE_SOFTMAX
    out_param = {
    }
    rnn = Rn.Rnn(hid_type, hid_param, hid_size, out_type, out_param)

    for rep in range(repeat):
        if verbose >= 1:
            print('Repeat: {} / {}'.format(rep, repeat))
        rnn.train(x, y, ep, window, stride, verbose)
        feat_size = y.shape[1]
        print('')
        current = 'BEGIN_SENTENCE'
        sentence_str = ''
        sentence = np.array([word2vector(current, feat_size, word2index)], dtype=np.int)
        for i in range(20):
            current = rnn.simulate(sentence)
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
            current = rnn.simulate(sentence)
            current = np.argmax(current[i])
            current = index2word[current]
            sentence = np.append(sentence, np.array([
                word2vector(current, feat_size, word2index)
            ]), axis=0)
            sentence_str = '{} {}'.format(sentence_str, current)
        print('{} '.format(sentence_str))
    save_model(rnn)


if __name__ == "__main__":
    main()

