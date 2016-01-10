#!/usr/bin/env python

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import wave
import os
import struct


def read_wave_file(filename, sample_width=2):
    w_file = wave.open(filename, 'rb')
    w_length = w_file.getnframes()
    print('{}: {}'.format(filename, w_file.getparams()))
    if sample_width != w_file.getsampwidth():
        print('  Invalid Sample width: {}'.format(w_file.getsampwidth()))
        return []
    w_dat = w_file.readframes(w_length)
    data = struct.unpack('<{}h'.format(w_length), w_dat)
    w_file.close()
    return data


def map_0_1(data):
    max_value = np.amax(data)
    min_value = np.amin(data)
    if min_value < 0:
        min_value = -min_value
    divider = float(max(max_value, min_value))
    ans = data / (2. * divider)
    ans += 0.5
    return ans


def create_data_file():
    # Load training wav files to np.array
    num_space = 100
    num_feature = 11
    sample_width = 2
    path = os.path.join('.', '__data_tmp__', 'numvoice', 'edit', 'train')
    train_data = np.array([])
    train_label = np.array([])
    for i in range(num_feature):
        filename = os.path.join(path, '{}.wav'.format(i))
        data = np.array(read_wave_file(filename, sample_width))
        train_data = np.append(train_data, data)
        label = np.zeros(((num_feature+1),), dtype=np.int)
        label[i] = 1
        label = np.tile(label, data.size)
        train_label = np.append(train_label, label)
        null_data = np.zeros((num_space,))
        train_data = np.append(train_data, null_data)
        null_label = np.zeros(((num_feature+1),), dtype=np.int)
        null_label[num_feature] = 1
        null_label = np.tile(null_label, num_space)
        train_label = np.append(train_label, null_label)
    train_label = np.reshape(train_label, (train_data.size, (num_feature+1)))
    train_data = np.reshape(train_data, (train_data.size, 1))
    train_data = map_0_1(train_data)

    # Load test wave files to np.array
    path = os.path.join('.', '__data_tmp__', 'numvoice', 'edit', 'test')
    test_dataset = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            data = np.array(read_wave_file(filename, sample_width))
            data = np.reshape(data, (data.size, 1))
            data = map_0_1(data)
            test_dataset.append(data)

    print('[{}, {}]'.format(np.amin(train_data), np.amax(train_data)))
    print(train_data.shape)
    print(train_data)
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
    fp.close()


def main():
    fp = h5.File('testdata02.h5', 'r')
    x = fp['/train/data'].value
    y = fp['/train/label'].value
    num_test = int(fp['/test/num'].value)
    test_x = []
    for i in range(num_test):
        test_x.append(fp['/test/data{}'.format(i)].value)
    fp.close()
    print(x)
    print(y)
    print(test_x)
    t = np.arange(x.shape[0])
    num_feature = y.shape[1] - 1
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.subplot(2, 1, 2)
    for i in range(num_feature):
        plt.plot(t, y[:, i], label='{}'.format(i))
    plt.legend(prop={'size': 7})
    plt.ylim(0, 1.1)
    plt.show()


if __name__ == "__main__":
    main()

