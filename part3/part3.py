import argparse
import sys
import numpy as np
from math import ceil
from time import time
from itertools import product
from scipy.stats import truncnorm

try:
    from img_to_mat.img_to_mat import img_to_mat, mat_to_img
except ImportError:
    print('run the following from the img_to_mat directory and try again:')
    print('python setup.py build_ext --inplace')

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


class NN:
    class Utils:
        def pad(x, ksize, strides, padding='SAME'):
            batch_size, in_height, in_width, in_channels = x.shape
            k_batch_size, k_height, k_width, k_channels = ksize
            if padding == 'SAME':
                out_height = ceil(float(in_height) / float(strides[1]))
                out_width = ceil(float(in_width) / float(strides[2]))
                if (in_height % strides[1] == 0):
                    pad_height = max(k_height - strides[1], 0)
                else:
                    pad_height = max(k_height - (in_height % strides[1]), 0)
                if (in_width % strides[2] == 0):
                    pad_width = max(k_width - strides[2], 0)
                else:
                    pad_width = max(k_width - (in_width % strides[2]), 0)
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
            elif padding == 'VALID':
                out_height = ceil(float(in_height - k_height + 1) / float(strides[1]))
                out_width = ceil(float(in_width - k_width + 1) / float(strides[2]))
                pad_top = 0
                pad_bottom = 0
                pad_left = 0
                pad_right = 0
            return pad_top, pad_bottom, pad_left, pad_right, out_height, out_width
        
        # def img_to_col(x, out_height, out_width, filter_height, filter_width, strides):
        #     batch_size = x.shape[0]
        #     in_channels = x.shape[3]
        #     x_col = np.zeros((batch_size * out_height * out_width, filter_height * filter_width * in_channels))
        #     for b in range(batch_size):
        #         for i in range(out_height):
        #             for j in range(out_width):
        #                 row = b * out_height * out_width + i * out_width + j
        #                 for p in range(filter_height):
        #                     for q in range(filter_width):
        #                         for r in range(in_channels):
        #                             col = p * filter_width * in_channels + q * in_channels + r
        #                             x_col[row, col] += x[b, strides[1] * i + p, strides[2] * j + q, r]
        #     return x_col
        
        # def col_to_img(x_col, out_height, out_width, in_height, in_width, filter_height, filter_width, strides):
        #     batch_size = x_col.shape[0] // (out_height * out_width)
        #     in_channels = x_col.shape[1] // (filter_height * filter_width)
        #     x = np.zeros((batch_size, in_height, in_width, in_channels))
        #     for b in range(batch_size):
        #         for i in range(out_height):
        #             for j in range(out_width):
        #                 row = b * out_height * out_width + i * out_width + j
        #                 for p in range(filter_height):
        #                     for q in range(filter_width):
        #                         for r in range(in_channels):
        #                             col = p * filter_width * in_channels + q * in_channels + r
        #                             x[b, strides[1] * i + p, strides[2] * j + q, r] += x_col[row, col]
        #     return x
        
        
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

            def out(self):
                self.out = self.z * self.mask
                return self.out

            def grad_z(self, grad_out):
                """Compute the derivative of the rectified linear unit using z."""
                self.grad_z = grad_out * self.mask
                return self.grad_z


        class softmax:
            def __init__(self, z):
                """Compute the softmax of z."""
                self.z = z
                self.e_z = np.exp(z - z.max())
                self.e_z_sum = self.e_z.reshape((self.e_z.shape[0], -1)).sum(axis = 1, keepdims = True)

            def out(self):
                self.out = self.e_z / self.e_z_sum
                return self.out
            
            def grad_z(self, grad_out):
                """Compute the derivative of the softmax of z."""
                # self.grad_z = (grad_out * self.out) - grad_out.dot(np.outer(self.out, self.out))
                self.grad_z = (grad_out - np.diagonal(grad_out.dot(self.out.T))) * self.out
                return self.grad_z
    
    
    class Layers:
        class flatten:
            def __init__(self, x):
                """Flatten tensor to shape [batch_size, x_height * x_width * in_channels]."""
                self.x = x

            def out(self):
                self.out_val = self.x.reshape((self.x.shape[0], -1))
                return self.out_val
            
            def grad_x(self, grad_out):
                self.grad_x_val = grad_out.reshape(self.x.shape)
                return self.grad_x_val
            
        
        class fullconn:
            def __init__(self, x, w):
                """Fully connected layer."""
                assert(x.shape[-1] == w.shape[0])
                self.x = x
                self.w = w
                self.batch_size = x.shape[0]
                self.in_channels = x.shape[1]
                self.out_channels = w.shape[1]

            def out(self):
                self.out_val = self.x.dot(self.w)
                return self.out_val

            def grad_x(self, grad_out):
                self.grad_x_val = grad_out.dot(self.w.T)
                return self.grad_x_val

            def grad_w(self, grad_out):
                self.grad_w_val = self.x.T.dot(grad_out)
                return self.grad_w_val
            
            
        class conv2d:
            def __init__(self, x, w, strides=[1, 1, 1, 1], padding='SAME'):
                assert(x.shape[3] == w.shape[2])
                self.x = x
                self.w = w
                self.strides = strides
                self.batch_size = x.shape[0]
                self.in_height = x.shape[1]
                self.in_width = x.shape[2]
                self.filter_height = w.shape[0]
                self.filter_width = w.shape[1]
                self.in_channels = w.shape[2]
                self.out_channels = w.shape[3]
                
                ksize = (1, self.filter_height, self.filter_width, 1)
                padding_out = NN.Utils.pad(x, ksize, strides, padding = padding)
                self.pad_top = padding_out[0]
                self.pad_bottom = padding_out[1]
                self.pad_left = padding_out[2]
                self.pad_right = padding_out[3]
                self.out_height = padding_out[4]
                self.out_width = padding_out[5]
                self.x_pad = np.pad(x, [(0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)], mode='constant')
            
            def out_bkp(self):
                self.out = np.empty((self.batch_size, self.out_height, self.out_width, self.out_channels))
                for b in range(self.batch_size):
                    for i in range(self.out_height):
                        for j in range(self.out_width):
                            for k in range(self.out_channels):
                                self.out[b, i, j, k] = (self.x_pad[b, (self.strides[1] * i):(self.strides[1] * i + self.filter_height), (self.strides[2] * j):(self.strides[2] * j + self.filter_width), :] * self.w[::-1, ::-1, :, k]).sum(axis = (0, 1))
                return self.out
            
            def out(self):
                # x_col = NN.Utils.img_to_col(self.x_pad, self.out_height, self.out_width, self.filter_height, self.filter_width, self.strides)
                x_mat = img_to_mat(self.x_pad, self.out_height, self.out_width, self.filter_height, self.filter_width, self.strides)
                w_mat = self.w[::-1, ::-1].reshape((-1, self.out_channels))
                self.fullconn = NN.Layers.fullconn(x_mat, w_mat)
                out_mat = self.fullconn.out()
                self.out_val = out_mat.reshape((self.batch_size, self.out_height, self.out_width, self.out_channels))
                return self.out_val
            
            def grad_x(self, grad_out):
                grad_out_reshaped = grad_out.reshape((-1, self.out_channels))
                grad_x_mat = self.fullconn.grad_x(grad_out_reshaped)
                # self.grad_x_val = NN.Utils.col_to_img(grad_x_col, self.out_height, self.out_width, self.x_pad.shape[1], self.x_pad.shape[2], self.filter_height, self.filter_width, self.strides)
                self.grad_x_val = mat_to_img(grad_x_mat, self.out_height, self.out_width, self.x_pad.shape[1], self.x_pad.shape[2], self.filter_height, self.filter_width, self.strides)
                if self.pad_top > 0:
                    self.grad_x_val = self.grad_x_val[:, self.pad_top:, :, :]
                if self.pad_bottom > 0:
                    self.grad_x_val = self.grad_x_val[:, :-self.pad_bottom, :, :]
                if self.pad_left > 0:
                    self.grad_x_val = self.grad_x_val[:, :, self.pad_left:, :]
                if self.pad_right > 0:
                    self.grad_x_val = self.grad_x_val[:, :, :-self.pad_right, :]
                return self.grad_x_val

            def grad_w(self, grad_out):
                grad_out_reshaped = grad_out.reshape((self.batch_size * self.out_height * self.out_width, self.out_channels))
                grad_w_col = self.fullconn.grad_w(grad_out_reshaped)
                self.grad_w_val = grad_w_col.reshape(self.w.shape)
                return self.grad_w_val


        class maxpool2d:
            def __init__(self, x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
                self.x = x
                self.ksize = ksize
                self.strides = strides
                self.batch_size = x.shape[0]
                self.in_height = x.shape[1]
                self.in_width = x.shape[2]
                self.in_channels = x.shape[3]
                self.out_channels = self.in_channels
                padding_out = NN.Utils.pad(x, ksize, strides, padding = padding)
                self.pad_top = padding_out[0]
                self.pad_bottom = padding_out[1]
                self.pad_left = padding_out[2]
                self.pad_right = padding_out[3]
                self.out_height = padding_out[4]
                self.out_width = padding_out[5]
                self.x_pad = np.pad(x, [(0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)], mode='constant')

            def out_bkp(self):
                self.out_val = np.empty((self.batch_size, self.out_height, self.out_width, self.out_channels))
                self.pos = np.zeros((self.batch_size, self.out_height, self.out_width, self.out_channels, 4), dtype = int)
                for b in range(self.batch_size):
                    for i in range(self.out_height):
                        stride_i = self.strides[1] * i
                        for j in range(self.out_width):
                            stride_j = self.strides[2] * j
                            for k in range(self.out_channels):
                                x_block = self.x_pad[b, (self.strides[1] * i):(self.strides[1] * i + self.ksize[1]), (self.strides[2] * j):(self.strides[2] * j + self.ksize[2]), k]
                                max_val = x_block.max()
                                self.out_val[b, i, j, k] = max_val
                                max_inds = list(np.unravel_index(x_block.argmax(), x_block.shape))
                                self.pos[b, i, j, k] = np.array([b, self.strides[1] * i + max_inds[0], self.strides[2] * j + max_inds[1], k])
                return self.out_val
            
            def grad_x_bkp(self, grad_out):
                self.grad_x_val = np.zeros(self.x_pad.shape)
                for b in range(self.batch_size):
                    for i in range(self.out_height):
                        for j in range(self.out_width):
                            for k in range(self.out_channels):
                                self.grad_x_val[tuple(self.pos[b, i, j, k])] = grad_out[b, i, j, k]
                return self.grad_x_val
            
            def out(self):
                x_reshaped = self.x_pad.transpose((3, 0, 1, 2)).reshape((self.in_channels * self.batch_size, self.in_height, self.in_width, 1))
                # x_col = NN.Utils.img_to_col(x_reshaped, self.out_height, self.out_width, self.ksize[1], self.ksize[2], self.strides)
                x_mat = img_to_mat(x_reshaped, self.out_height, self.out_width, self.ksize[1], self.ksize[2], self.strides)
                self.x_mat_maxcols = x_mat.argmax(axis = 1)
                out_reshaped = x_mat[np.arange(x_mat.shape[0]), self.x_mat_maxcols]
                self.out_val = out_reshaped.reshape((self.in_channels, self.batch_size, self.out_height, self.out_width)).transpose((1, 2, 3, 0))
                return self.out_val
            
            def grad_x(self, grad_out):
                grad_x_mat = np.zeros((self.in_channels * self.batch_size * self.out_height * self.out_width, self.ksize[1] * self.ksize[2]))
                grad_x_mat[np.arange(grad_x_mat.shape[0]), self.x_mat_maxcols] = grad_out.transpose((3, 0, 1, 2)).flatten()
                # grad_x_reshaped = NN.Utils.col_to_img(grad_x_mat, self.out_height, self.out_width, self.x_pad.shape[1], self.x_pad.shape[2], self.ksize[1], self.ksize[2], self.strides)
                grad_x_reshaped = mat_to_img(grad_x_mat, self.out_height, self.out_width, self.x_pad.shape[1], self.x_pad.shape[2], self.ksize[1], self.ksize[2], self.strides)
                if self.pad_top > 0:
                    grad_x_reshaped = grad_x_reshaped[:, self.pad_top:, :, :]
                if self.pad_bottom > 0:
                    grad_x_reshaped = grad_x_reshaped[:, :-self.pad_bottom, :, :]
                if self.pad_left > 0:
                    grad_x_reshaped = grad_x_reshaped[:, :, self.pad_left:, :]
                if self.pad_right > 0:
                    grad_x_reshaped = grad_x_reshaped[:, :, :-self.pad_right, :]
                self.grad_x_val = grad_x_reshaped.reshape((self.in_channels, self.batch_size, self.in_height, self.in_width)).transpose((1, 2, 3, 0))
                return self.grad_x_val
    
    
    class Cost:
        class mean_squared_error:
            def __init__(self, y_label, y_out):
                self.y_label = y_label
                self.y_out = y_out
            
            def out(self):
                self.out_val = 0.5 * (self.y_label - self.y_out) ** 2
                return self.out_val
            
            def grad_x(self):
                self.grad_x_val = self.y_out - self.y_label
                return self.grad_x_val


        class cross_entropy:
            def __init__(self, y_label, y_out):
                self.y_label = y_label
                self.y_out = y_out
            
            def out(self):
                self.out_val = -np.sum(self.y_label * np.log(self.y_out), axis = 1)
                return self.out_val
            
            def grad_x(self):
                self.grad_x_val = -self.y_label / self.y_out
                return self.grad_x_val


        class softmax_cross_entropy:
            def __init__(self, y_label, y_out):
                self.y_label = y_label
                self.y_out = y_out
            
            def out(self):
                softmax = NN.Activations.softmax(self.y_out)
                self.out_val = -np.sum(self.y_label * np.log(softmax.out()), axis = 1)
                return self.out_val
            
            def grad_z(self):
                self.grad_z_val = self.y_out - self.y_label
                return self.grad_z_val
    
    
    class Metrics:
        def accuracy(y_label, y_out):
            correct_prediction = np.equal(np.argmax(y_label, axis = 1), np.argmax(y_out, axis = 1))
            return correct_prediction.astype(np.float32).mean()


    class Optimizers:
        class Adagrad:
            def __init__(self, lr, n_layers, initial_accumulator_value=0.1):
                self.lr = lr
                self.dw_sum_sq = [initial_accumulator_value for i in range(n_layers)]
                self.db_sum_sq = [initial_accumulator_value for i in range(n_layers)]

            def step(self, dw, db):
                for i in range(len(dw)):
                    if not dw[i] is None:
                        self.dw_sum_sq[i] += dw[i] ** 2
                    if not db[i] is None:
                        self.db_sum_sq[i] += db[i] ** 2

            def lr_eff(self):
                for i in range(len(self.dw_sum_sq)):
                    lr_eff_w = self.lr / np.sqrt(self.dw_sum_sq[i])
                    lr_eff_b = self.lr / np.sqrt(self.db_sum_sq[i])
                return lr_eff_w, lr_eff_b


class Model:
    def __init__(self, x_image_shape=[28, 28], lr=0.5, optimizer=None, cost=NN.Cost.mean_squared_error):
        self.started = False
        self.epochs_completed = 0
        self.x_image_shape = x_image_shape
        self.lr = lr
        self.optimizer_func = optimizer
        self.cost_func = cost
        self.w = []
        self.dw = []
        self.b = []
        self.db = []
        self.f = []
        self.s = []

    def optimize_with(self, optimizer_func, lr=None):
        self.optimizer_func = optimizer_func
        if lr != self.lr:
            self.lr = lr
        return self

    def minimize(self, cost_func):
        self.cost_func = cost_func
        return self

    def init_layer_vars(self, w=None, b=None):
        if not self.started:
            self.w.append(w)
            self.dw.append(None)
            self.b.append(b)
            self.db.append(None)
            self.f.append(None)
            self.s.append(None)
        
    def eval(self, batch_x_images):
        # Layer 0 (2D convolution layer)
        filter_shape_0 = [10, 10]
        in_channels_0 = 1
        out_channels_0 = 32

        # Reshape input
        x_0 = np.reshape(batch_x_images, [batch_x_images.shape[0]] + list(self.x_image_shape) + [in_channels_0])

        self.init_layer_vars(w = NN.Variables.weight_variable(filter_shape_0 + [in_channels_0, out_channels_0]),
                             b = NN.Variables.bias_variable([out_channels_0]))

        self.f[0] = NN.Layers.conv2d(x_0, self.w[0])
        z_0 = self.f[0].out() + self.b[0]
        self.s[0] = NN.Activations.relu(z_0)

        y_0 = self.s[0].out()

        # Layer 1 (Max pool 2x2)
        in_channels_1 = out_channels_0
        out_channels_1 = in_channels_1
        
        self.init_layer_vars()

        self.f[1] = NN.Layers.maxpool2d(y_0)

        x_1 = self.f[1].out()

        # Layer 2 (2D convolution layer)
        filter_shape_2 = [5, 5]
        in_channels_2 = out_channels_1
        out_channels_2 = 16

        self.init_layer_vars(w = NN.Variables.weight_variable(filter_shape_2 + [in_channels_2, out_channels_2]),
                             b = NN.Variables.bias_variable([out_channels_2]))

        self.f[2] = NN.Layers.conv2d(x_1, self.w[2])
        z_2 = self.f[2].out() + self.b[2]
        self.s[2] = NN.Activations.relu(z_2)

        x_2 = self.s[2].out()

        # Layer 3 (Max pool 2x2)
        in_channels_3 = out_channels_2
        out_channels_3 = in_channels_3
        
        self.init_layer_vars()

        self.f[3] = NN.Layers.maxpool2d(x_2)

        x_3 = self.f[3].out()

        # Layer 4 (Flatten)
        in_channels_4 = out_channels_3
        out_channels_4 = in_channels_4
        
        self.init_layer_vars()

        self.f[4] = NN.Layers.flatten(x_3)

        x_4 = self.f[4].out()

        # Layer 5 (Fully connected layer)
        in_channels_5 = x_4.shape[-1]
        out_channels_5 = 1024

        self.init_layer_vars(w = NN.Variables.weight_variable([in_channels_5, out_channels_5]),
                             b = NN.Variables.bias_variable([out_channels_5]))

        self.f[5] = NN.Layers.fullconn(x_4, self.w[5])
        z_5 = self.f[5].out() + self.b[5]
        self.s[5] = NN.Activations.relu(z_5)

        x_5 = self.s[5].out()

        # Layer 6 (Fully connected layer)
        in_channels_6 = out_channels_5
        out_channels_6 = 10

        self.init_layer_vars(w = NN.Variables.weight_variable([in_channels_6, out_channels_6]),
                             b = NN.Variables.bias_variable([out_channels_6]))

        self.f[6] = NN.Layers.fullconn(x_5, self.w[6])
        z_6 = self.f[6].out() + self.b[6]
        self.s[6] = NN.Activations.softmax(z_6)

        x_6 = self.s[6].out()
        
        self.batch_y_out = x_6
        
        return self.batch_y_out
    
    def backpropagate(self, batch_y_labels):
        n_layers = len(self.dw)
        if not self.started:
            if not self.optimizer_func is None:
                self.optimizer = self.optimizer_func(self.lr, n_layers)
            self.started = True

        cost = self.cost_func(batch_y_labels, self.batch_y_out)

        if self.optimizer_func is None:
            lr_eff_w, lr_eff_b = self.lr, self.lr
        else:
            lr_eff_w, lr_eff_b = self.optimizer.lr_eff()

        if 'grad_z' in vars(self.cost_func):
            delta = -cost.grad_z()
        elif 'grad_x' in vars(self.cost_func):
            delta = -self.s[n_layers - 1].grad_z(cost.grad_x())
        else:
            raise Exception("No cost function recognized.")

        if not self.dw[n_layers - 1] is None:
            self.db[n_layers - 1] = lr_eff_b * np.mean(np.sum(delta, axis = tuple(range(1, delta.ndim - 1))), axis = 0)
            self.b[n_layers - 1] += self.db[n_layers - 1]
            self.dw[n_layers - 1] = lr_eff_w * np.mean(self.f[n_layers - 1].grad_w(delta), axis = 0)
            self.w[n_layers - 1] += self.dw[n_layers - 1]
        
        for i in reversed(range(n_layers - 1)):
            if self.dw[i] is None:
                delta = self.f[i + 1].grad_x(delta)
            else:
                delta = self.s[i].grad_z(self.f[i + 1].grad_x(delta))
                self.db[i] = lr_eff_b * np.mean(np.sum(delta, axis = tuple(range(delta.ndim - 1))), axis = 0)
                self.b[i] += self.db[i]
                self.dw[i] = lr_eff_w * np.mean(self.f[i].grad_w(delta), axis = 0)
                self.w[i] += self.dw[i]

        if not self.optimizer_func is None:
            self.optimizer.step(self.dw, self.db)

    def train_step(self, batch_x_images, batch_y_labels, timing=False):
        if timing:
            start_forward_time = time()
        batch_y_out = self.eval(batch_x_images)
        if timing:
            forward_time = (time() - start_forward_time) / batch_size
            start_backward_time = time()
        self.backpropagate(batch_y_labels, lr = lr)
        if timing:
            backward_time = (time() - start_backward_time) / batch_size

def main(_):
    # Open output files
    train_stream = open("train_accuracy.csv", "w")
    print("Epoch,Train Accuracy\n", file=train_stream)

    test_stream = open("test_accuracy.csv", "w")
    print("Epoch,Test Accuracy\n", file=test_stream)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    model = Model(mnist.train).optimize_with(NN.Optimizers.Adagrad, lr = 1e-3).minimize(NN.Cost.softmax_cross_entropy)
    timing = True

    # Train
    batch_size = 100
    epochs = 2
    batch_num = 1
    while mnist.train.epochs_completed < epochs:
        prev_nepoch = mnist.train.epochs_completed
        batch_x_images, batch_y_labels = mnist.train.next_batch(batch_size)
        if mnist.train.epochs_completed > prev_nepoch:
            train_accuracy = NN.Metrics.accuracy(mnist.train.images, mnist.train.labels)
            print("{},{}\n".format(prev_nepoch, train_accuracy), file=train_stream)
            test_accuracy = NN.Metrics.accuracy(mnist.test.images, mnist.test.labels)
            print("{},{}\n".format(prev_nepoch, test_accuracy), file=test_stream)
            batch_num = 1
        model.train_step(batch_x_images, batch_y_labels, timing=timing)
        if batch_num % 100 == 0:
            batch_accuracy = NN.Metrics.accuracy(batch_y_labels, batch_y_out)
            if timing:
                print("Epoch: {0}, Batch: {1}, Accuracy: {2:.4f}, Forward Time: {3:.2f}, Backward Time: {4:.2f}".format(mnist.train.epochs_completed + 1, batch_num, batch_accuracy, forward_time, backward_time))
            else:
                print("Epoch: {0}, Batch: {1}, Accuracy: {2:.4f}".format(mnist.train.epochs_completed + 1, batch_num, batch_accuracy))
        batch_num += 1

    test_accuracy = NN.Metrics.accuracy(mnist.test.labels, model.eval(mnist.test.images))
    print("\n Test Accuracy: {0:.4f}".format(test_accuracy))

    train_stream.close()
    test_stream.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)