from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='cuda_ba_det',
    ext_modules=[
        CUDAExtension('cuda_ba_det',
                      sources=['devo/fastba/ba.cpp', 'devo/fastba/ba_cuda_det.cu'],
                      extra_compile_args={
                          'cxx':  ['-O3'], 
                          'nvcc': ['-O3'],
                      }),
    ],
    cmdclass={'build_ext': BuildExtension}
)
