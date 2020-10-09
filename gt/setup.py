from distutils.core import setup, Extension


cpp_args = ['-std=c++17', "-fopenmp"]

ext_modules = [
    Extension(
    'cgt',
        ['cgt.cpp'],
        include_dirs=['pybind11/include'],
    language='c++',
    extra_compile_args=cpp_args,
    ),
]

setup(
    name='cgt',
    version='0.0.1',
    author='Xinyan DAI',
    author_email='xinyan.dai@outlook.com',
    description='cget',
    ext_modules=ext_modules,
)