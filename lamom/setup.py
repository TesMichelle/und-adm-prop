import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import scipy


extra_compile_args = [""]#["-stdlib=libc", "-std=c"]


ext = Extension("calcmom", ["calculate_moments.pyx"],
                include_dirs = [numpy.get_include()],
                language='c',
                extra_compile_args=extra_compile_args,
      )

setup(ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})
