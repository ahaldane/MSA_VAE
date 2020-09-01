#!/usr/bin/env python
from numpy.distutils.core import setup, Extension

# compile me with
# python ./setup.py build_ext --inplace

module1 = Extension('highmarg',
                    sources = ['highmarg.c', 'art.c'],
                    extra_compile_args = ['-O3', '-Wall'])
                    #extra_compile_args = ['-g -Wall'])

setup (name = 'highmarg',
       version = '1.0',
       description = 'higher order marginal calculator',
       ext_modules = [module1])
