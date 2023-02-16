#!/usr/bin/env python3
from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        Extension("skinny", ["skinny.pyx"]),
        Extension("_util", ["_util.pyx"]),
        Extension("lin_util", ["lin_util.pyx"]),
        Extension("cnf_util", ["cnf_util.pyx"]),
    ]),
)
