import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))



setup(
    name='devo',
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        #CUDAExtension('ba_cpu',
        #    sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cpu.cpp', 'devo/fastba/reproject.cu'],
        #    extra_compile_args={
        #        'cxx':  ['-O3'], 
        #        'nvcc': ['-O3'],
        #    }),
        CUDAExtension('ba_cpu_profile',
            sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cpu_profile.cpp', 'devo/fastba/reproject.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        #CUDAExtension('ba_cpu_fp128',
        #    sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cpu_fp128.cpp', 'devo/fastba/reproject.cu'],
        #    extra_compile_args={
        #        'cxx':  ['-O3', '-fext-numeric-literals'],
        #        'nvcc': ['-O3'],
        #    },
        #    libraries=['quadmath']
        #    ),
        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

