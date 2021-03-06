# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


# Weight and bias

def weight_variable(shape):
    """
    Initialize weight matrices with truncated normal distribution with standard deviation of 0.1.

    In general, shape = [filter_height, filter_width, in_channels, out_channels].
    """
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    """
    Initialize bias vectors with constant value of 0.1.

    In general, shape = [out_channels].
    """
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


# Layers

def conv2d(x, W):
    """2D convolution of x with filter W, unit stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    """2x2 max pooling layer with non-overlapping kernel stride."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def flatten(x):
    size = 1
    for dim in x.shape.as_list()[1:]:
        size *= dim
    return tf.reshape(x, [-1, size])

def fullconn(x, W):
    return tf.matmul(x, W)


# Model

def model(x_image, x_image_shape = [28, 28]):
    # Layer 0 (2D convolution layer)
    filter_shape_0 = [10, 10]
    in_channels_0 = 1
    out_channels_0 = 32

    # Reshape input
    batch = -1
    x_0 = tf.reshape(x_image, [batch] + list(x_image_shape) + [in_channels_0])

    w_0 = weight_variable(filter_shape_0 + [in_channels_0, out_channels_0])
    b_0 = bias_variable([out_channels_0])

    z_0 = conv2d(x_0, w_0) + b_0

    x_1 = tf.nn.relu(z_0)

    # Layer 1 (Max pool 2x2)
    in_channels_1 = out_channels_0
    out_channels_1 = in_channels_1
        
    x_2 = maxpool2d(x_1)

    # Layer 2 (2D convolution layer)
    filter_shape_2 = [5, 5]
    in_channels_2 = out_channels_1
    out_channels_2 = 16

    w_2 = weight_variable(filter_shape_2 + [in_channels_2, out_channels_2])
    b_2 = bias_variable([out_channels_2])

    z_2 = conv2d(x_2, w_2) + b_2

    x_3 = tf.nn.relu(z_2)

    # Layer 3 (Max pool 2x2)
    in_channels_3 = out_channels_2
    out_channels_3 = in_channels_3
        
    x_4 = maxpool2d(x_3)

    # Layer 4 (Flatten)
    in_channels_4 = out_channels_3
    out_channels_4 = in_channels_4

    x_5 = flatten(x_4)

    # Layer 5 (Fully connected layer)
    in_channels_5 = x_5.shape.as_list()[-1]
    out_channels_5 = 1024

    w_5 = weight_variable([in_channels_5, out_channels_5])
    b_5 = bias_variable([out_channels_5])

    z_5 = fullconn(x_5, w_5) + b_5

    x_6 = tf.nn.relu(z_5)

    # Layer 6 (Fully connected layer)
    in_channels_6 = out_channels_5
    out_channels_6 = 10

    w_6 = weight_variable([in_channels_6, out_channels_6])
    b_6 = bias_variable([out_channels_6])

    z_6 = fullconn(x_6, w_6) + b_6

    return z_6


def main(_):
    # Open output files
    train_stream = open("train_accuracy.csv", "w")
    print("Epoch,Train Accuracy\n", file=train_stream)

    test_stream = open("test_accuracy.csv", "w")
    print("Epoch,Test Accuracy\n", file=test_stream)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y = model(x)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdagradOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Train
    batch_size = 100
    epochs = 200
    batch_num = 1
    while mnist.train.epochs_completed < epochs:
        prev_nepoch = mnist.train.epochs_completed
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        if mnist.train.epochs_completed > prev_nepoch:
            train_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
            print("{},{}\n".format(prev_nepoch, train_accuracy), file=train_stream)
            test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print("{},{}\n".format(prev_nepoch, test_accuracy), file=test_stream)
            batch_num = 1
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if batch_num % 100 == 0:
            batch_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            print("Epoch: {0}, Batch: {1}, Accuracy: {2:.4f}".format(mnist.train.epochs_completed + 1, batch_num, batch_accuracy))
        batch_num += 1

    # Test trained model
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("\n Test Accuracy: {0:.4f}".format(test_accuracy))

    train_stream.close()
    test_stream.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)