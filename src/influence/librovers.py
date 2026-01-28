"""
TODO: Gaurav
Auto generated with bash.
Create a shared library with make and link to it for production.
"""

# hacked bindings
# lazy evaluation
# JIT

import cppyy
import os
import platform

# os.environ["EXTRA_CLING_ARGS"] = "-g"

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
source_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, "cpp"))
include_dir = os.path.join(source_dir, 'include')
libs_dir = os.path.join(source_dir, 'libs')

# include paths
cppyy.add_include_path(include_dir)
cppyy.add_include_path(libs_dir)

# Header for rover environment. T
# This file includes everything neccessary for the rover domain
cppyy.include(os.path.join(include_dir, 'rover_domain/environment.hpp'))

# making c++ namespaces visible
rover_domain = cppyy.gbl.rover_domain
std = cppyy.gbl.std

# cppyy.set_debug()

# This will give us a python traceback for C++ segfaults
# import cppyy.ll
# cppyy.ll.set_signals_as_exception(True)
