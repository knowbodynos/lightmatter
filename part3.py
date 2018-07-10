import argparse
import sys
import numpy as np
from math import ceil
from itertools import product
from scipy.stats import truncnorm

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


class NN:
    class Variables:
        def weight_variable(shape):
            """
            Initialize weight matrices with truncated normal distribution with standard deviation of 0.1.

            In general, shape = [filter_height, filter_width, in_channels, out_channels].
            """
            stddev = 0.1
            size = 1
            for dim in shape:
                size *= dim
            return truncnorm.rvs(-2 * stddev, 2 * stddev, size=size).reshape(shape)

        def bias_variable(shape):
            """
            Initialize bias vectors with constant value of 0.1.

            In general, shape = [out_channels].
            """
            const = 0.1
            return np.full(shape, const)
    
    
    class Activations:
        class relu:
            def __init__(self, z):
                """Compute the rectified linear unit using z."""
                self.z = z
                self.mask_nonneg = (z >= 0).astype(np.float32)
                mask_zero = (z == 0).astype(np.float32)
                self.mask = self.mask_nonneg - 0.5 * mask_zero

            def eval(self):
                return self.z * self.mask

            def ddz(self):
                """Compute the derivative of the rectified linear unit using z."""
                result = np.zeros(list(self.z.shape) * 2, dtype = np.float32)
                for z_inds in product(*[range(i) for i in self.z.shape]):
                    result[z_inds][z_inds] = self.mask[z_inds]
                return result


        class softmax:
            def __init__(self, z):
                """Compute the softmax of z."""
                self.z = z
                self.e_z = np.exp(z)
                print(z)
                print(self.e_z)

            def eval(self):
                return self.e_z / self.e_z.sum()

            def ddz(self):
                """Compute the derivative of the softmax of z."""
                softmax_z = self.eval()
                result = np.empty(list(self.z.shape) * 2, dtype = np.float32)
                for z_inds in product(*[range(i) for i in self.z.shape]):
                    result[z_inds] = softmax_z[z_inds] * softmax_z
                    result[z_inds][z_inds] = softmax_z[z_inds] - result[z_inds][z_inds]
                return result
    
    
    class Layers:
        class conv2d:
            def __init__(self, x, w, strides=[1, 1, 1], padding='SAME'):
                assert(x.shape[2] == w.shape[2])
                self.x = x
                self.w = w
                self.strides = strides
                self.in_height = x.shape[0]
                self.in_width = x.shape[1]
                self.filter_height = w.shape[0]
                self.filter_width = w.shape[1]
                self.in_channels = w.shape[2]
                self.out_channels = w.shape[3]
                if padding == 'SAME':
                    self.out_height = ceil(float(self.in_height) / float(strides[0]))
                    self.out_width = ceil(float(self.in_width) / float(strides[1]))
                    if (self.in_height % strides[0] == 0):
                        pad_height = max(self.filter_height - strides[0], 0)
                    else:
                        pad_height = max(self.filter_height - (self.in_height % strides[0]), 0)
                    if (self.in_width % strides[1] == 0):
                        pad_width = max(self.filter_width - strides[1], 0)
                    else:
                        pad_width = max(self.filter_width - (self.in_width % strides[1]), 0)
                    self.pad_top = pad_height // 2
                    self.pad_bottom = pad_height - self.pad_top
                    self.pad_left = pad_width // 2
                    self.pad_right = pad_width - self.pad_left
                elif padding == 'VALID':
                    self.out_height = ceil(float(self.in_height - self.filter_height + 1) / float(strides[0]))
                    self.out_width = ceil(float(self.in_width - self.filter_width + 1) / float(strides[1]))
                    self.pad_top = 0
                    self.pad_bottom = 0
                    self.pad_left = 0
                    self.pad_right = 0
                self.x_pad = np.pad(x, [(self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0,0)], mode='constant')
                self.y = np.empty([self.out_height, self.out_width, self.out_channels], dtype = x.dtype)

            def eval(self):
                self.y = np.empty(self.y.shape, dtype = self.x.dtype)
                for i in range(self.out_height):
                    stride_i = self.strides[0] * i
                    for j in range(self.out_width):
                        stride_j = self.strides[1] * j
                        for k in range(self.out_channels):
                            self.y[i, j, k] = (self.x_pad[stride_i:stride_i + self.filter_height, stride_j:stride_j + self.filter_width, :] * self.w[::-1, ::-1, :, k]).sum()
                return self.y

            def ddx(self):
                result = np.zeros(self.y.shape + self.x_pad.shape, dtype = np.float32)
                for inds in product(*[range(i) for i in self.y.shape]):
                    min_ind_0 = self.strides[0] * inds[0]
                    max_ind_0 = min_ind_0 + self.filter_height
                    min_ind_1 = self.strides[1] * inds[1]
                    max_ind_1 = min_ind_1 + self.filter_width
                    result[inds][min_ind_0:max_ind_0, min_ind_1:max_ind_1, :] = self.w[(self.filter_height - 1)::-1, (self.filter_width - 1)::-1, :, inds[2]]
                return result

            def ddw(self):
                result = np.zeros(self.y.shape + self.w.shape, dtype = np.float32)
                for inds in product(*[range(i) for i in self.y.shape]):
                    min_ind_0 = self.strides[0] * inds[0]
                    max_ind_0 = min_ind_0 + self.filter_height
                    min_ind_1 = self.strides[1] * inds[1]
                    max_ind_1 = min_ind_1 + self.filter_width
                    result[inds][(self.filter_height - 1)::-1, (self.filter_width - 1)::-1, :, inds[2]] = self.x_pad[min_ind_0:max_ind_0, min_ind_1:max_ind_1, :]
                return result


        class maxpool2d:
            def __init__(self, x, ksize=[2, 2, 1], strides=[2, 2, 1], padding='SAME'):
                self.x = x
                self.ksize = ksize
                self.strides = strides
                self.in_height = x.shape[0]
                self.in_width = x.shape[1]
                self.in_channels = x.shape[2]
                self.out_channels = self.in_channels
                if padding == 'SAME':
                    self.out_height = ceil(float(self.in_height) / float(strides[0]))
                    self.out_width = ceil(float(self.in_width) / float(strides[1]))
                    if (self.in_height % strides[0] == 0):
                        pad_height = max(ksize[0] - strides[0], 0)
                    else:
                        pad_height = max(ksize[0] - (self.in_height % strides[0]), 0)
                    if (self.in_width % strides[1] == 0):
                        pad_width = max(ksize[1] - strides[1], 0)
                    else:
                        pad_width = max(ksize[1] - (self.in_width % strides[1]), 0)
                    self.pad_top = pad_height // 2
                    self.pad_bottom = pad_height - self.pad_top
                    self.pad_left = pad_width // 2
                    self.pad_right = pad_width - self.pad_left
                elif padding == 'VALID':
                    self.out_height = ceil(float(self.in_height - ksize[1] + 1) / float(strides[0]))
                    self.out_width = ceil(float(self.in_width - ksize[2] + 1) / float(strides[1]))
                    self.pad_top = 0
                    self.pad_bottom = 0
                    self.pad_left = 0
                    self.pad_right = 0
                self.x_pad = np.pad(x, [(self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0,0)], mode='constant')
                self.y = np.empty([self.out_height, self.out_width, self.out_channels], dtype = x.dtype)
                self.pos = np.zeros(list(self.y.shape) + [3], dtype = int)

            def eval(self):
                self.y = np.empty(self.y.shape, dtype = self.x.dtype)
                self.pos = np.zeros(list(self.y.shape) + [3], dtype = int)
                for i in range(self.out_height):
                    stride_i = self.strides[0] * i
                    for j in range(self.out_width):
                        stride_j = self.strides[1] * j
                        for k in range(self.out_channels):
                            x_block = self.x_pad[stride_i:stride_i + self.ksize[0], stride_j:stride_j + self.ksize[1], k]
                            max_val = x_block.max()
                            self.y[i, j, k] = max_val
                            max_inds = list(np.unravel_index(x_block.argmax(), x_block.shape))
                            self.pos[i, j, k] = np.array([stride_i + max_inds[0], stride_j + max_inds[1], k])
                return self.y

            def unpool(self, x_pooled):
                result = np.zeros(self.x.shape, dtype = self.x.dtype)
                for i in range(self.out_height):
                    stride_i = self.strides[0] * i
                    for j in range(self.out_width):
                        stride_j = self.strides[1] * j
                        for k in range(self.out_channels):
                            result[tuple(self.pos[i, j, k])] = x_pooled[i, j, k]
                return result


        class flatten:
            def __init__(self, x):
                """Flatten tensor to shape [batch_size, x_height * x_width * in_channels]."""
                self.x = x
                size = 1
                for dim in x.shape:
                    size *= dim
                self.y = np.empty([size], dtype = x.dtype)

            def eval(self):
                self.y = self.x.reshape((self.y.size))
                return self.y
            
            def unflatten(self, x_flattened):
                return x_flattened.reshape(self.x.shape)


        class fullconn:
            def __init__(self, x, w):
                """Fully connected layer."""
                assert(x.size == w.shape[0])
                self.x = x
                self.w = w
                self.in_channels = x.size
                self.out_channels = w.shape[1]

            def eval(self):
                return np.matmul(self.x, self.w)

            def ddx(self):
                return self.w.T

            def ddw(self):
                result = np.zeros([self.out_channels, self.in_channels, self.out_channels], dtype = self.x.dtype)
                for i in range(self.out_channels):
                    result[i, :, i] = self.x
                return result
    class Cost:
        class cross_entropy:
            def __init__(self, y_label, y_out):
                self.y_label = y_label
                self.y_out = y_out
            
            def eval(self):
                return -np.sum(self.y_label * np.log(self.y_out))
            
            def ddx(self):
                return -self.y_label / self.y_out

        class softmax_cross_entropy:
            def __init__(self, y_label, y_out):
                self.y_label = y_label
                self.y_out = y_out
            
            def eval(self):
                softmax_out = NN.Activations.softmax(self.y_out)
                return -np.sum(self.y_label * np.log(softmax_out.eval))
            
            def ddz(self):
                return self.y_out - self.y_label
    
    
    class Metrics:
        def accuracy(y_label, y_out):
            return (np.argmax(y_label) == np.argmax(y_out)).astype(np.float32)


class Model:
    def __init__(self, x_images, y_labels, n_layers, x_image_shape=[28, 28], seed=None):
        np.random.seed(seed = seed)
        self.x_images = x_images
        self.y_labels = y_labels
        self.n_layers = n_layers
        self.i = 0
        self.epochs_completed = 0
        self.x_image_shape = x_image_shape
        self.w = [None for i in range(n_layers)]
        self.b = [None for i in range(n_layers)]
        self.f = [None for i in range(n_layers)]
        self.g = [None for i in range(n_layers)]
        self.h = [None for i in range(n_layers)]
        self.s = [None for i in range(n_layers)]
        self.delta = [None for i in range(n_layers)]
        
    def eval(self):
        # Layer 0 (2D convolution layer)
        filter_shape_0 = [5, 5]
        in_channels_0 = 1
        out_channels_0 = 32
        
        # Reshape input
        x_image = self.x_images[self.i]
        x_input = np.reshape(x_image, list(self.x_image_shape) + [in_channels_0])

        if self.w[0] is None:
            self.w[0] = NN.Variables.weight_variable(filter_shape_0 + [in_channels_0, out_channels_0])
            self.b[0] = NN.Variables.bias_variable([out_channels_0])

        self.f[0] = NN.Layers.conv2d(x_input, self.w[0])
        z_0 = self.f[0].eval() + self.b[0]
        self.s[0] = NN.Activations.relu(z_0)

        #  Max pool
        self.g[0] = NN.Layers.maxpool2d(self.s[0].eval())
        y_0 = self.g[0].eval()

        # Layer 1 (2D convolution layer)
        filter_shape_1 = [5, 5]
        in_channels_1 = out_channels_0
        out_channels_1 = 64

        if self.w[1] is None:
            self.w[1] = NN.Variables.weight_variable(filter_shape_1 + [in_channels_1, out_channels_1])
            self.b[1] = NN.Variables.bias_variable([out_channels_1])

        self.f[1] = NN.Layers.conv2d(y_0, self.w[1])
        z_1 = self.f[1].eval() + self.b[1]
        self.s[1] = NN.Activations.relu(z_1)

        #  Max pool
        self.g[1] = NN.Layers.maxpool2d(self.s[1].eval())

        #  Flatten
        self.h[1] = NN.Layers.flatten(self.g[1].eval())
        y_1 = self.h[1].eval()

        # Layer 2 (Fully connected layer)
        in_channels_2 = y_1.shape[-1]
        out_channels_2 = 1024

        if self.w[2] is None:
            self.w[2] = NN.Variables.weight_variable([in_channels_2, out_channels_2])
            self.b[2] = NN.Variables.bias_variable([out_channels_2])

        self.f[2] = NN.Layers.fullconn(y_1, self.w[2])
        z_2 = self.f[2].eval() + self.b[2]
        self.s[2] = NN.Activations.relu(z_2)

        y_2 = self.s[2].eval()

        # Layer 3 (Fully connected layer)
        in_channels_3 = out_channels_2
        out_channels_3 = 10

        if self.w[3] is None:
            self.w[3] = NN.Variables.weight_variable([in_channels_3, out_channels_3])
            self.b[3] = NN.Variables.bias_variable([out_channels_3])

        self.f[3] = NN.Layers.fullconn(y_2, self.w[3])
        self.z_out = self.f[3].eval() + self.b[3]
        self.s[3] = NN.Activations.softmax(self.z_out)

        self.y_out = self.s[3].eval()
        return self.y_out
    
    def compute_deltas(self, lr = 0.5):
        cost = NN.Cost.softmax_cross_entropy(self.y_labels[self.i], self.z_out)
        delta = cost.ddz()
        self.delta[-1] = -lr * np.tensordot(delta, self.s[-1].ddz(), axes = len(self.y_out.shape))
        for i in reversed(range(1, self.n_layers)):
            intermed = np.tensordot(self.delta[i], self.f[i].ddx(), axes = len(self.delta[i].shape))
            if not self.h[i - 1] is None:
                intermed = self.h[i - 1].unflatten(intermed)
            if not self.g[i - 1] is None:
                intermed = self.g[i - 1].unpool(intermed)
            self.delta[i - 1] = np.tensordot(intermed, self.s[i - 1].ddz(), axes = len(intermed.shape))
        
#         intermed = self.g[-3].unpool(self.h[-3].unflatten(np.tensordot(dz[-2], self.f[-2].ddx(), axes = len(dz[-2].shape))))
#         dz[-3] = np.tensordot(intermed, self.s[-3].ddz(), axes = len(intermed.shape))
        
#         intermed = self.g[-4].unpool(np.tensordot(dz[-3], self.f[-3].ddx(), axes = len(dz[-3].shape)))
#         dz[-4] = np.tensordot(intermed, self.s[-4].ddz(), axes = len(intermed.shape))
    
    def compute_dw(self):
        dw = []
        for i in range(self.n_layers):
            dw.append(np.tensordot(self.delta[i], self.f[i].ddw(), axes = len(self.delta[i].shape)))
        return dw
#         self.dw_3 = np.tensordot(self.dz_3, self.f_3.ddw(), axes = len(self.dz_3.shape))
#         self.dw_2 = np.tensordot(self.dz_2, self.f_2.ddw(), axes = len(self.dz_2.shape))
#         self.dw_1 = np.tensordot(self.dz_1, self.f_1.ddw(), axes = len(self.dz_1.shape))
#         self.dw_0 = np.tensordot(self.dz_0, self.f_0.ddw(), axes = len(self.dz_0.shape))
    
    def compute_db(self):
        db = []
        for i in range(self.n_layers):
            db.append(self.delta[i].mean(axis = tuple(range(len(self.delta[i].shape) - 1))))
        return db
#         self.db_3 = self.dz_3.sum(axis = range(len(self.dz_3.shape) - 1))
#         self.db_2 = self.dz_2.sum(axis = range(len(self.dz_2.shape) - 1))
#         self.db_1 = self.dz_1.sum(axis = range(len(self.dz_1.shape) - 1))
#         self.db_0 = self.dz_0.sum(axis = range(len(self.dz_0.shape) - 1))
    
    def batch(self, batch_size, lr = 0.5):
        y_labels_batch = []
        y_out_batch = []
        for count in range(batch_size):
            if self.i == len(self.x_images):
                data_inds = np.random.permutation(self.x_images.shape[0])
                self.x_images = self.x_images[data_inds]
                self.y_labels = self.y_labels[data_inds]
                self.i = 0
                self.epochs_completed += 1
            y_labels_batch.append(self.y_labels[self.i])
            y_out_batch.append(self.eval())
            self.compute_deltas(lr = lr)
            step_dw = self.compute_dw()
            step_db = self.compute_db()
            for j in range(self.n_layers):
                self.w[j] += (1 / batch_size) * step_dw[j]
                self.b[j] += (1 / batch_size) * step_db[j]
            self.i += 1
        return y_labels_batch, y_out_batch
    
    def train(self, batch_size, epochs, lr = 0.5):
        batch_num = 1
        while self.epochs_completed < epochs:
            prev_nepoch = self.epochs_completed
            y_labels_batch, y_out_batch = self.batch(batch_size, lr = 0.5)
            if self.epochs_completed > prev_nepoch:
                batch_num = 1
            if batch_num % 50 == 0:
                batch_accuracy = sum(map(NN.Metrics.accuracy, y_labels_batch, y_out_batch)) / batch_size
                print("Epoch: {0}, Batch: {1}, Accuracy: {2:.4f}".format(self.epochs_completed + 1, batch_num, batch_accuracy))
            batch_num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    model = Model(mnist.train.images, mnist.train.labels, 4)
    model.train(3, 3)

    test_accuracy = sum(map(NN.Metrics.accuracy, mnist.test.images, mnist.test.labels)) / mnist.test.images.shape[0]
    print("\n Test Accuracy: {0:.4f}".format(test_accuracy))

    # batch_accuracy = sum(map(NN.Metrics.accuracy, *model.batch(2, lr=1e-3))) / 2
    # print("\n Accuracy: {0:.4f}".format(batch_accuracy))