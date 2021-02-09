#!/usr/bin/env python3
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("RASM_tool.pyx"))
# setup(ext_modules=cythonize("RASM_tool.pyx"))

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext


# from setuptools import Extension, setup
# from Cython.Build import cythonize
# i

# setup(
#     cmdclass = {'build_ext': build_ext},
#     ext_modules = [Extension("bresenhamLine",
#                              sources=["bresenhamLine.pyx", "c_bresenhamLine.c"],
#                              include_dirs=[numpy.get_include()])],
# )

# setup(
#     cmdclass = {'build_ext': build_ext},
#     ext_modules = [Extension("dwa_evaluation",
#                              sources=["dwa_evaluation.pyx", "c_dwa_evaluation.c"],
#                              include_dirs=[numpy.get_include()])],
# )

# setup(
#     cmdclass = {'build_ext': build_ext},
#     ext_modules = [Extension("RASM_tools",
#                              sources=["RASM_tools.pyx", "cpp_RASM_tools.cpp"],
#                              language="c++",
#                              include_dirs=[numpy.get_include()])],
# )