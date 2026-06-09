cmake_minimum_required(VERSION 3.20)
project(cuda_preprocess CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Standalone shared library (usable without Python/pybind11)
add_library(cuda_preprocess_lib SHARED
    preprocess.cpp
    preprocess_kernel.cu
)

set_target_properties(cuda_preprocess_lib PROPERTIES
    CUDA_ARCHITECTURES "80;86;89"   # Ampere + Ada; adjust per GPU
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(cuda_preprocess_lib PRIVATE CUDA::cudart)

# Optional: standalone benchmark executable
add_executable(preprocess_bench benchmark_main.cpp)
target_link_libraries(preprocess_bench PRIVATE cuda_preprocess_lib CUDA::cudart)
