import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))



setup(
    name='devo',
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        #CUDAExtension('cuda_ba_red',
        #    sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_red.cu'],
        #    extra_compile_args={
        #        'cxx':  ['-O3'], 
        #        'nvcc': ['-O3'],
        #    }),
        #CUDAExtension('cuda_ba_kahan',
        #    sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_red_kahan.cu'],
        #    extra_compile_args={
        #        'cxx':  ['-O0'], 
        #        'nvcc': ['-O0'],
        #    }),
        CUDAExtension('cuda_ba_red2',
            sources=['devo/fastba/ba2.cpp', 'devo/fastba/ba_cuda_red2.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

