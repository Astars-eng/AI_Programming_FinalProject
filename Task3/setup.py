from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name='cuda_net',
    zip_safe=False,
    ext_modules=[
        CUDAExtension(
            name='cuda_net',
            sources=['pybind.cpp', 'cnn_module.cu'],
            include_dirs=[
            os.path.abspath(os.path.dirname(__file__)),
                ],
            library_dirs=['/usr/local/cuda/lib64'],
            libraries=['cudart', 'curand','cublas'], 
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: C++',
    ],
)