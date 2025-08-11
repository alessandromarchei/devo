import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

ROOT = osp.dirname(osp.abspath(__file__))



setup(
    name='devo',
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr_debug',
           sources=['devo/altcorr/correlation.cpp', 'devo/altcorr/correlation_kernel_debug.cu'],
           extra_compile_args={
               'cxx':  ['-O3'], 
               'nvcc': ['-O3'],
           }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

