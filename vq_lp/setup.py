from distutils.core import setup, Extension


cpp_args = ['-std=c++17', "-fopenmp"]

ext_modules = [
    Extension(
    'cpp_vq_lp',
        ['vq_lp.cpp'],
        include_dirs=['pybind11/include'],
    language='c++',
    extra_compile_args=cpp_args,
    ),
]

setup(
    name='cpp_vq_lp',
    version='0.0.1',
    author='Xinyan DAI',
    author_email='xinyan.dai@outlook.com',
    description='cpp_vq_lp',
    ext_modules=ext_modules,
)