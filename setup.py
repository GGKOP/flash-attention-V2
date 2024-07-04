from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention_extension',
    ext_modules=[
        CUDAExtension('flash_attention_forward', [
            './ops/forward.cpp',
            './ops/flash_attention_forward.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)