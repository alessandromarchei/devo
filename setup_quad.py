import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths
ROOT = osp.dirname(osp.abspath(__file__))


setup(
    name='quad_test',
    ext_modules=[
        CppExtension(
            'quad_test',
            sources=['devo/fastba/quad_test.cpp'],
            include_dirs=include_paths(),  # ensures torch/torch.h and friends
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-fext-numeric-literals', '-D_GLIBCXX_USE_CXX11_ABI=0'],
            },
            libraries=['quadmath'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
