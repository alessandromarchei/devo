import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))



setup(
    name='devo',
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_ba_double',
            sources=['devo/fastba/ba_double.cpp', 'devo/fastba/ba_cuda_double.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

