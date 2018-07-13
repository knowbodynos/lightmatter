import numpy as np
cimport numpy as np
cimport cython

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def img_to_mat(np.ndarray[DTYPE_t, ndim=4] x, int out_height, int out_width, int filter_height, int filter_width, list strides):
    cdef int batch_size = x.shape[0]
    cdef int in_channels = x.shape[3]
    cdef np.ndarray[DTYPE_t, ndim=2] x_mat = np.zeros((batch_size * out_height * out_width, filter_height * filter_width * in_channels), dtype=x.dtype)
    img_to_mat_cython(x_mat, x, batch_size, out_height, out_width, filter_height, filter_width, in_channels, strides)
    return x_mat


@cython.boundscheck(False)
cdef int img_to_mat_cython(np.ndarray[DTYPE_t, ndim=2] x_mat, np.ndarray[DTYPE_t, ndim=4] x, int batch_size, int out_height, int out_width, int filter_height, int filter_width, int in_channels, list strides) except? -1:
    cdef int b, i, j, row, p, q, r, col
    for b in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                row = b * out_height * out_width + i * out_width + j
                for p in range(filter_height):
                    for q in range(filter_width):
                        for r in range(in_channels):
                            col = p * filter_width * in_channels + q * in_channels + r
                            x_mat[row, col] += x[b, strides[1] * i + p, strides[2] * j + q, r]


def mat_to_img(np.ndarray[DTYPE_t, ndim=2] x_mat, int out_height, int out_width, int in_height, int in_width, int filter_height, int filter_width, list strides):
    cdef int batch_size = x_mat.shape[0] // (out_height * out_width)
    cdef int in_channels = x_mat.shape[1] // (filter_height * filter_width)
    cdef np.ndarray[DTYPE_t, ndim=4] x = np.zeros((batch_size, in_height, in_width, in_channels), dtype=x_mat.dtype)
    mat_to_img_cython(x, x_mat, batch_size, out_height, out_width, filter_height, filter_width, in_channels, strides)
    return x


@cython.boundscheck(False)
cdef int mat_to_img_cython(np.ndarray[DTYPE_t, ndim=4] x, np.ndarray[DTYPE_t, ndim=2] x_mat, int batch_size, int out_height, int out_width, int filter_height, int filter_width, int in_channels, list strides) except? -1:
    cdef int b, i, j, row, p, q, r, col
    for b in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                row = b * out_height * out_width + i * out_width + j
                for p in range(filter_height):
                    for q in range(filter_width):
                        for r in range(in_channels):
                            col = p * filter_width * in_channels + q * in_channels + r
                            x[b, strides[1] * i + p, strides[2] * j + q, r] += x_mat[row, col] 