# Jetson Backend - TensorRT + DLA
# Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

find_package(CUDAToolkit REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/local/tensorrt/include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/local/tensorrt/lib)
find_library(TENSORRT_PARSERS nvparsers
    HINTS /usr/lib/aarch64-linux-gnu /usr/local/tensorrt/lib)
find_library(TENSORRT_ONNXPARSER nvonnxparser
    HINTS /usr/lib/aarch64-linux-gnu /usr/local/tensorrt/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found")
endif()

# Sources
set(JETSON_SOURCES
    jetson_engine.cpp
    trt_builder.cpp
    dla_utils.cpp
)

# Create shared library
add_library(_jetson SHARED
    ${JETSON_SOURCES}
    bindings.cpp
)

target_include_directories(_jetson PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/common
    ${TENSORRT_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(_jetson PRIVATE
    mast3r_common
    pybind11::module
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PARSERS}
    ${TENSORRT_ONNXPARSER}
    CUDA::cudart
)

set_target_properties(_jetson PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Install
install(TARGETS _jetson LIBRARY DESTINATION mast3r_runtime/backends)
