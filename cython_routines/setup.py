from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line

ext_modules = [Extension("cython_functions", ["cython_functions.pyx"],
                         include_dirs = [np.get_include()])]

setup(
  name = 'Cython functions for mppy',
  cmdclass = {'build_ext': build_ext},         # <---- New line
  ext_modules = ext_modules
)