from distutils.core import setup, Extension
import numpy

setup(ext_modules=[Extension("_C_matrix_ops",
      sources=["matrix_ops.c", "C_matrix_ops.i"],
      include_dirs=[numpy.get_include()])])
