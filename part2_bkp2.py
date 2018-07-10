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

def conv_2d(x, W):
    """2D convolution of x with filter W, unit stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """2x2 max pooling layer with non-overlapping kernel stride."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def flatten(x):
    size = 1
    for dim in x.shape.as_list()[1:]:
        size *= dim
    return tf.reshape(x, [-1, size])

def full_conn(x, W):
    return tf.matmul(x, W)


# Model

def model(x_image, x_image_shape = [28, 28]):
    # Initialize input
    batch_size = -1
    in_channels = 1
    x_input = tf.reshape(x_image, [batch_size] + x_image_shape + [in_channels])

    # Layer 1 (2D convolution layer)
    filter_shape_1 = [10, 10]
    in_channels_1 = in_channels
    out_channels_1 = 32

    W_conv2d_1 = weight_variable(filter_shape_1 + [in_channels_1, out_channels_1])
    b_conv2d_1 = bias_variable([out_channels_1])

    y_conv2d_1 = tf.nn.relu(conv_2d(x_input, W_conv2d_1) + b_conv2d_1)
    y_pool2x2_1 = max_pool_2x2(y_conv2d_1)

    # Layer 2 (2D convolution layer)
    filter_shape_2 = [5, 5]
    in_channels_2 = out_channels_1
    out_channels_2 = 16

    W_conv2d_2 = weight_variable(filter_shape_2 + [in_channels_2, out_channels_2])
    b_conv2d_2 = bias_variable([out_channels_2])

    y_conv2d_2 = tf.nn.relu(conv_2d(y_pool2x2_1, W_conv2d_2) + b_conv2d_2)
    y_pool2x2_2 = max_pool_2x2(y_conv2d_2)

    # Layer 3 (Fully connected layer)
    y_flattened = flatten(y_pool2x2_2)

    in_channels_3 = y_flattened.shape.as_list()[-1]
    out_channels_3 = 1024

    W_fullconn_3 = weight_variable([in_channels_3, out_channels_3])
    b_fullconn_3 = bias_variable([out_channels_3])

    y_fullconn_3 = tf.nn.relu(full_conn(y_flattened, W_fullconn_3) + b_fullconn_3)

    # Layer 4 (Fully connected layer)
    in_channels_4 = out_channels_3
    out_channels_4 = 10

    W_fullconn_4 = weight_variable([in_channels_4, out_channels_4])
    b_fullconn_4 = bias_variable([out_channels_4])

    y_fullconn_4 = full_conn(y_fullconn_3, W_fullconn_4) + b_fullconn_4

    return y_fullconn_4


def main(_):
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
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    train_step = tf.train.AdagradOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)