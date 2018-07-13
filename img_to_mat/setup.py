from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('img_to_mat', ['img_to_mat.pyx'], include_dirs = [numpy.get_include()]),
]

setup(
    ext_modules = cythonize(extensions),
)
