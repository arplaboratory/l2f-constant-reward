from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys, os
import setuptools

absolute_path = os.path.dirname(os.path.abspath(__file__))

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the `get_include()`
    method can be invoked."""

    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'l2f',
        ['src/bindings/l2f.cpp'],
        include_dirs=[
            get_pybind_include(),
            os.path.join(absolute_path, "include"),
            os.path.join(absolute_path, "external", "rl_tools", "include"),
            os.path.join(absolute_path, "external", "rl_tools", "external", "json", "include"),
        ],
        language='c++',
        extra_compile_args=['-std=c++17'] if not sys.platform.startswith('win') else ['/std:c++17'],
        define_macros=[('RL_TOOLS_ENABLE_JSON', None)]
    ),
]

print("include_dirs: ", ext_modules[0].include_dirs)

setup(
    name='l2f',
    version='0.0.1',
    author='Jonas Eschmann',
    description='Learning to Fly in Seconds',
    long_description='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
