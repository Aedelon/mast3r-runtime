# CUDA Backend - NVIDIA GPU with cuBLAS
# Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

find_package(CUDAToolkit REQUIRED)

# CUDA sources
set(CUDA_SOURCES
    cuda_engine.cu
    preprocessing.cu
    attention.cu
    matching.cu
)

# Create shared library
add_library(_cuda SHARED
    ${CUDA_SOURCES}
    bindings.cpp
)

target_include_directories(_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/common
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(_cuda PRIVATE
    mast3r_common
    pybind11::module
    CUDA::cudart
    CUDA::cublas
)

# CUDA specific flags
set_target_properties(_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"  # Volta to Hopper
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Install
install(TARGETS _cuda LIBRARY DESTINATION mast3r_runtime/backends)