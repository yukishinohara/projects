from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def get_gaussian_random_values(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def get_constant_values(shape, value):
  return tf.Variable(tf.constant(value, shape=shape))

def conv2d(in_vector, filter_size=3, in_channel_num=3, out_channel_num=3):
  return tf.nn.conv2d(in_vector,
                      get_gaussian_random_values([filter_size, 
                                                  filter_size,
                                                  in_channel_num,
                                                  out_channel_num]),
                      strides=[1,1,1,1],
                      padding='SAME')

def max_pool(in_vector, asize=2):
  return tf.nn.max_pool(in_vector,
                        ksize=[1,asize,asize,1],
                        strides=[1,asize,asize,1],
                        padding='SAME')

def relu_conv2d(in_vector, filter_size=3, in_channel_num=3, out_channel_num=3):
  return tf.nn.relu(conv2d(in_vector, filter_size, in_channel_num, out_channel_num) +
                    get_constant_values([out_channel_num], 0.1))

def relu_fully_connected(in_vector, in_node_num, out_node_num):
  return tf.nn.relu(tf.matmul(in_vector,
                              get_gaussian_random_values([in_node_num, out_node_num])) +
                    get_constant_values([out_node_num], 0.1))

def softmax_fully_connected(in_vector, in_node_num, out_node_num):
  return tf.nn.softmax(tf.matmul(in_vector,
                              get_gaussian_random_values([in_node_num, out_node_num])) +
                    get_constant_values([out_node_num], 0.1))


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  x = tf.placeholder(tf.float32, [None, 784])
  x_tensor = tf.reshape(x, [-1,28,28,1])

  # Define a CNN model
  #
  #  x[28x28x1]==conv(5x5x32)==>h1[28x28x12]==maxpool(2x2)==>h2[14x14x12]
  #  ==conv(5x5x16)==>h3[14x14x16]==maxpool(2x2)==>h4[7x7x16]
  #  ==fullcn(784x10)==>y[10]
  h1 = relu_conv2d(x_tensor, 5, 1, 12)
  h2 = max_pool(h1, 2)
  h3 = relu_conv2d(h2, 5, 12, 16)
  h4 = max_pool(h3, 2)
  h4_flat = tf.reshape(h4, [-1,7*7*16])
  h5 = relu_fully_connected(h4_flat, 7*7*16, 1024)
  y  = softmax_fully_connected(h5, 1024, 10)
  
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Prepare to get its accuracy
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Start a Tensorflow session
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Train the model
  for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # Do not print unless i%100=0
    if i%100 != 0:
      continue
    # Print current accuracy
    print('result[{0}] = {1}'.format(i,
                                     sess.run(accuracy, feed_dict={
                                              x : batch_xs,
                                              y_: batch_ys})))
  print('TOTAL = {0}'.format(sess.run(accuracy, feed_dict={
                                      x : mnist.test.images,
                                      y_: mnist.test.labels})))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='__mnist_data_downloaded',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


