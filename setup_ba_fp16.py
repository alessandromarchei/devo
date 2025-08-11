from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='devo',
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        # CUDAExtension(
        #     'cuda_ba_fp16_chol',
        #     sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_fp16_chol.cu', 'devo/fastba/reproject.cu'],
        #     extra_compile_args={
        #         'cxx': ['-O3'],
        #         'nvcc': [
        #             '-O3',
        #             '-gencode=arch=compute_89,code=sm_89',
        #             '-gencode=arch=compute_89,code=compute_89',
        #         ]
        #     }
        # ),
        # CUDAExtension(
        #     'cuda_ba_fp32_chol',
        #     sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_fp32_chol.cu', 'devo/fastba/reproject.cu'],
        #     extra_compile_args={
        #         'cxx': ['-O3'],
        #         'nvcc': [
        #             '-O3',
        #             '-gencode=arch=compute_89,code=sm_89',
        #             '-gencode=arch=compute_89,code=compute_89',
        #         ]
        #     }
        # ),
        # CUDAExtension(
        #     'cuda_ba_fp16_lu',
        #     sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_fp16_lu.cu', 'devo/fastba/reproject.cu'],
        #     extra_compile_args={
        #         'cxx': ['-O3'],
        #         'nvcc': [
        #             '-O3',
        #             '-gencode=arch=compute_89,code=sm_89',
        #             '-gencode=arch=compute_89,code=compute_89',
        #         ]
        #     }
        # ),
        CUDAExtension(
            'cuda_ba_bf16_chol',
            sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_bf16_chol.cu', 'devo/fastba/reproject.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_89,code=sm_89',
                    '-gencode=arch=compute_89,code=compute_89',
                ]
            }
        ),
        # CUDAExtension(
        #     'cuda_ba_bf16_lu',
        #     sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_bf16_lu.cu', 'devo/fastba/reproject.cu'],
        #     extra_compile_args={
        #         'cxx': ['-O3'],
        #         'nvcc': [
        #             '-O3',
        #             '-gencode=arch=compute_89,code=sm_89',
        #             '-gencode=arch=compute_89,code=compute_89',
        #         ]
        #     }
        # ),
        # CUDAExtension(
        #     'cuda_ba_fp32_lu',
        #     sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_fp32_lu.cu', 'devo/fastba/reproject.cu'],
        #     extra_compile_args={
        #         'cxx': ['-O3'],
        #         'nvcc': [
        #             '-O3',
        #             '-gencode=arch=compute_89,code=sm_89',
        #             '-gencode=arch=compute_89,code=compute_89',
        #         ]
        #     }
        # ),

        # CUDAExtension(
        #     'cuda_ba_fp32_chol2',
        #     sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_fp32_chol2.cu', 'devo/fastba/reproject.cu'],
        #     extra_compile_args={
        #         'cxx': ['-O3'],
        #         'nvcc': [
        #             '-O1',
        #             '-gencode=arch=compute_89,code=sm_89',
        #             '-gencode=arch=compute_89,code=compute_89',
        #         ]
        #     }
        # )
    ],
    cmdclass={'build_ext': BuildExtension}
)
