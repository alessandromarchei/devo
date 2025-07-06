from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='ba_cpu',
    ext_modules=[
        CppExtension('ba_cpu', 
                     sources=['devo/fastba/ba_cpu.cpp'],
                     extra_compile_args={'cxx': ['-O3']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
