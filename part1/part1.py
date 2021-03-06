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
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

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
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    # Train
    batch_size = 100
    epochs = 2
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