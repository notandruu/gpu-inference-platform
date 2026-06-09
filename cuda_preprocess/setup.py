"""Build the CUDA preprocessing Python extension with pybind11."""
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="cuda_preprocess",
    ext_modules=[
        CUDAExtension(
            name="cuda_preprocess",
            sources=[
                "bindings.cpp",
                "preprocess.cpp",
                "preprocess_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_80",   # Ampere (A100/RTX 30xx); adjust per GPU
                    "--extended-lambda",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
