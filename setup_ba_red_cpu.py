import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))



setup(
    name='devo',
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_ba_red_cpu_fw',
            sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_red_cpu_fw.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba_red_cpu_bw',
            sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_red_cpu_bw.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba_red_cpu_fw_save',
            sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_red_cpu_fw_save.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('cuda_ba_red_cpu_bw_save',
            sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_red_cpu_bw_save.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
    
        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

