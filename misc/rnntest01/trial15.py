#!/usr/bin/env python

from __future__ import print_function
import scipy
from scipy.io import wavfile
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import os


def read_wave_file(filename, sample_width=2, stride=1, stride_wave=1):
    _, data2 = scipy.io.wavfile.read(filename)
    ceps, _, _ = mfcc(data2, nceps=13, nframeparam=stride)

    return np.array(ceps), np.array(data2[::stride_wave])


def map_0_1(data):
    data[np.isinf(data)] = 0.
    data[np.isneginf(data)] = 0.
    data[np.isnan(data)] = 0.
    ans = np.array(data, dtype=np.float)
    for col in range(ans.shape[1]):
        max_value = np.amax(data[:, col])
        min_value = np.amin(data[:, col])
        if min_value < 0:
            min_value = -min_value
        divider = float(max(max_value, min_value))
        ans[:, col] = (data[:, col] * 0.5) / divider
        ans[:, col] += 0.5
    return ans


def activate_ceps(data):
    data[np.isinf(data)] = 0.
    data[np.isneginf(data)] = 0.
    data[np.isnan(data)] = 0.
    data = np.array(data, dtype=np.float)
    for col in range(data.shape[1]):
        max_value = np.amax(data[:, col])
        min_value = np.amin(data[:, col])
        divider = float(max_value - min_value)
        data[:, col] /= divider
        offset = np.mean(data[:, col])
        data[:, col] += 0.5 - offset
    return data


def create_data_file():
    # Load training wav files to np.array
    num_feature = 11
    sample_width = 2
    stride = 30
    stride_wave = 30
    num_space = 50
    num_space_wave = 50
    path = os.path.join('.', '__data_tmp__', 'numvoice', 'edit', 'train')
    train_data = None
    train_wave = None
    train_label = None
    for i in range(num_feature):
        filename = os.path.join(path, '{}.wav'.format(i))
        data, wav = read_wave_file(filename, sample_width, stride, stride_wave)
        data_row, data_col = data.shape
        null_data = np.zeros((num_space, data_col))
        null_wav = np.zeros((num_space_wave,))
        if train_data is None:
            train_data = np.append(data, null_data, axis=0)
            train_wave = np.append(wav, null_wav)
        else:
            train_data = np.append(train_data,
                                   np.append(data, null_data, axis=0),
                                   axis=0)
            train_wave = np.append(train_wave,
                                   np.append(wav, null_wav))

        label = np.zeros(((num_feature+1),), dtype=np.int)
        label[i] = 1
        label = np.tile(label, data_row)
        label = np.reshape(label, (data_row, num_feature+1))
        null_label = np.zeros(((num_feature+1),), dtype=np.int)
        null_label[num_feature] = 1
        null_label = np.tile(null_label, num_space)
        null_label = np.reshape(null_label, (num_space, num_feature+1))
        if train_label is None:
            train_label = np.append(label, null_label, axis=0)
        else:
            train_label = np.append(train_label,
                                    np.append(label, null_label, axis=0),
                                    axis=0)
    train_data = activate_ceps(train_data)
    train_wave = np.reshape(train_wave, (train_wave.size, 1))
    train_wave = map_0_1(train_wave)

    # Load test wave files to np.array
    path = os.path.join('.', '__data_tmp__', 'numvoice', 'edit', 'test')
    test_dataset = []
    test_wavset = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            data, wav = read_wave_file(filename, sample_width, stride, stride_wave)
            data = activate_ceps(data)
            wav = np.reshape(wav, (wav.size, 1))
            wav = map_0_1(wav)
            test_dataset.append(data)
            test_wavset.append(wav)

    print('[{}, {}]'.format(np.amin(train_data), np.amax(train_data)))
    print(train_data.shape)
    print(train_data)
    print('[{}, {}]'.format(np.amin(train_wave), np.amax(train_wave)))
    print(train_wave.shape)
    print(train_wave)
    print(train_label.shape)
    print(train_label)
    for i in range(len(test_dataset)):
        print('[{}, {}]'.format(np.amin(test_dataset[i]), np.amax(test_dataset[i])))
        print(test_dataset[i].shape)
        print(test_dataset[i])

    # Save them to the sample file
    output_file = os.path.join('.', 'testdata02.h5')
    fp = h5.File(output_file, 'w')
    train_h5 = fp.create_group('train')
    train_h5.create_dataset('data',
                            shape=train_data.shape,
                            dtype=train_data.dtype,
                            data=train_data,
                            compression="gzip")
    train_h5.create_dataset('wave',
                            shape=train_wave.shape,
                            dtype=train_wave.dtype,
                            data=train_wave,
                            compression="gzip")
    train_h5.create_dataset('label',
                            shape=train_label.shape,
                            dtype=train_label.dtype,
                            data=train_label,
                            compression="gzip")
    test_h5 = fp.create_group('test')
    test_h5.create_dataset('num',
                           shape=(1,),
                           dtype=np.int,
                           data=np.array([len(test_dataset)]))
    for i in range(len(test_dataset)):
        test_h5.create_dataset('data{}'.format(i),
                               shape=test_dataset[i].shape,
                               dtype=test_dataset[i].dtype,
                               data=test_dataset[i],
                               compression="gzip")
        test_h5.create_dataset('wave{}'.format(i),
                               shape=test_wavset[i].shape,
                               dtype=test_wavset[i].dtype,
                               data=test_wavset[i],
                               compression="gzip")
    fp.close()


def create_data_file2():
    # Load training wav files to np.array
    num_feature = 5
    sample_width = 2
    stride = 2
    stride_wave = 2
    num_space = 50
    num_space_wave = 50
    path = os.path.join('.', '__data_tmp__', 'aeiou', 'train')
    train_data = None
    train_wave = None
    train_label = None
    for i in range(num_feature):
        filename = os.path.join(path, '{}.wav'.format(i))
        data, wav = read_wave_file(filename, sample_width, stride, stride_wave)
        data_row, data_col = data.shape
        null_data = np.zeros((num_space, data_col))
        null_wav = np.zeros((num_space_wave,))
        if train_data is None:
            train_data = np.append(data, null_data, axis=0)
            train_wave = np.append(wav, null_wav)
        else:
            train_data = np.append(train_data,
                                   np.append(data, null_data, axis=0),
                                   axis=0)
            train_wave = np.append(train_wave,
                                   np.append(wav, null_wav))

        label = np.zeros(((num_feature+1),), dtype=np.int)
        label[i] = 1
        label = np.tile(label, data_row)
        label = np.reshape(label, (data_row, num_feature+1))
        null_label = np.zeros(((num_feature+1),), dtype=np.int)
        null_label[num_feature] = 1
        null_label = np.tile(null_label, num_space)
        null_label = np.reshape(null_label, (num_space, num_feature+1))
        if train_label is None:
            train_label = np.append(label, null_label, axis=0)
        else:
            train_label = np.append(train_label,
                                    np.append(label, null_label, axis=0),
                                    axis=0)
    train_data = activate_ceps(train_data)
    train_wave = np.reshape(train_wave, (train_wave.size, 1))
    train_wave = map_0_1(train_wave)

    # Load test wave files to np.array
    test_dataset = []
    test_wavset = []
    test_labelset = []
    for i in range(3):
        path = os.path.join('.', '__data_tmp__', 'aeiou', 'test', '{}'.format(i))
        test_data = test_wav = test_label = None
        for root, dirs, files in os.walk(path):
            for file in files:
                filename = os.path.join(root, file)
                labelstr = os.path.basename(filename)
                labelstr = (labelstr.split("_")[1]).split(".")[0]
                data, wav = read_wave_file(filename, sample_width, stride, stride_wave)
                data_row, _ = data.shape
                if test_data is None:
                    test_data = data
                    test_wav = wav
                else:
                    test_data = np.append(test_data, data, axis=0)
                    test_wav = np.append(test_wav, wav, axis=0)
                label = np.zeros(((num_feature+1),), dtype=np.int)
                label[int(labelstr)] = 1
                label = np.tile(label, data_row)
                label = np.reshape(label, (data_row, num_feature+1))
                if test_label is None:
                    test_label = label
                else:
                    test_label = np.append(test_label, label, axis=0)
        test_data = activate_ceps(test_data)
        test_wav = np.reshape(test_wav, (test_wav.size, 1))
        test_wav = map_0_1(test_wav)
        test_dataset.append(test_data)
        test_wavset.append(test_wav)
        test_labelset.append(test_label)

    print('[{}, {}]'.format(np.amin(train_data), np.amax(train_data)))
    print(train_data.shape)
    print(train_data)
    print('[{}, {}]'.format(np.amin(train_wave), np.amax(train_wave)))
    print(train_wave.shape)
    print(train_wave)
    print(train_label.shape)
    print(train_label)
    for i in range(len(test_dataset)):
        print('[{}, {}]'.format(np.amin(test_dataset[i]), np.amax(test_dataset[i])))
        print(test_dataset[i].shape)
        print(test_dataset[i])
        print(test_labelset[i].shape)
        print(test_labelset[i])

    # Save them to the sample file
    output_file = os.path.join('.', 'testdata03.h5')
    fp = h5.File(output_file, 'w')
    train_h5 = fp.create_group('train')
    train_h5.create_dataset('data',
                            shape=train_data.shape,
                            dtype=train_data.dtype,
                            data=train_data,
                            compression="gzip")
    train_h5.create_dataset('wave',
                            shape=train_wave.shape,
                            dtype=train_wave.dtype,
                            data=train_wave,
                            compression="gzip")
    train_h5.create_dataset('label',
                            shape=train_label.shape,
                            dtype=train_label.dtype,
                            data=train_label,
                            compression="gzip")
    test_h5 = fp.create_group('test')
    test_h5.create_dataset('num',
                           shape=(1,),
                           dtype=np.int,
                           data=np.array([len(test_dataset)]))
    for i in range(len(test_dataset)):
        test_h5.create_dataset('data{}'.format(i),
                               shape=test_dataset[i].shape,
                               dtype=test_dataset[i].dtype,
                               data=test_dataset[i],
                               compression="gzip")
        test_h5.create_dataset('wave{}'.format(i),
                               shape=test_wavset[i].shape,
                               dtype=test_wavset[i].dtype,
                               data=test_wavset[i],
                               compression="gzip")
        test_h5.create_dataset('label{}'.format(i),
                               shape=test_labelset[i].shape,
                               dtype=test_labelset[i].dtype,
                               data=test_labelset[i],
                               compression="gzip")
    fp.close()



def main():
    fp = h5.File('testdata03.h5', 'r')
    x = fp['/train/data'].value
    w = fp['/train/wave'].value
    y = fp['/train/label'].value
    num_test = int(fp['/test/num'].value)
    test_x = []
    for i in range(num_test):
        test_x.append(fp['/test/data{}'.format(i)].value)
    fp.close()
    print(w.shape)
    print(x.shape)
    print(y.shape)
    t = np.arange(w.shape[0])
    t2 = np.arange(x.shape[0])
    num_feature = y.shape[1] - 1
    plt.subplot(3, 1, 1)
    plt.plot(t, w)
    plt.subplot(3, 1, 2)
    plt.plot(t2, x)
    plt.subplot(3, 1, 3)
    kana = ['a', 'i', 'u', 'e', 'o']
    iro = ['#ff0000', '#ffff00', '#ff00ff', '#00ff00', '#00ffff', '#0000ff'
           '#770000', '#007700', '#000077', '#aaff00', '#000000']
    for i in range(num_feature):
        plt.plot(t2, y[:, i], label='{}'.format(kana[i]), color='{}'.format(iro[i]))
    plt.legend(prop={'size': 7})
    plt.ylim(0, 1.1)
    plt.show()


if __name__ == "__main__":
    main()

