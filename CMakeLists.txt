# MASt3R Runtime - Cross-platform build
# Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

cmake_minimum_required(VERSION 3.21)
project(mast3r_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options
option(BUILD_CPU "Build CPU backend" ON)
option(BUILD_CUDA "Build CUDA backend" OFF)
option(BUILD_MPS "Build MPS backend (macOS 15+ with MPSGraph SDPA)" OFF)
option(BUILD_JETSON "Build Jetson backend" OFF)
option(BUILD_TESTS "Build tests" OFF)

# Auto-detect platform
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(BUILD_MPS ON CACHE BOOL "MPS backend" FORCE)
    message(STATUS "[mast3r] macOS arm64 detected -> MPS backend enabled")
endif()

if(NOT APPLE AND CMAKE_CUDA_COMPILER)
    set(BUILD_CUDA ON CACHE BOOL "CUDA backend" FORCE)
    message(STATUS "[mast3r] CUDA detected -> CUDA backend enabled")
endif()

# Check for Jetson (aarch64 Linux with CUDA)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    if(EXISTS "/etc/nv_tegra_release")
        set(BUILD_JETSON ON CACHE BOOL "Jetson backend" FORCE)
        message(STATUS "[mast3r] Jetson detected -> Jetson backend enabled")
    endif()
endif()

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "[mast3r] pybind11 found: ${pybind11_VERSION}")

# Common library
add_subdirectory(csrc/common)

# Platform-specific backends
if(BUILD_CPU)
    message(STATUS "[mast3r] Building CPU backend")
    add_subdirectory(csrc/cpu)
endif()

if(BUILD_MPS)
    message(STATUS "[mast3r] Building MPS backend (requires macOS 15+)")
    enable_language(OBJCXX)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
    add_subdirectory(csrc/mps)
endif()

if(BUILD_CUDA)
    message(STATUS "[mast3r] Building CUDA backend")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_subdirectory(csrc/cuda)
endif()

if(BUILD_JETSON)
    message(STATUS "[mast3r] Building Jetson backend")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # TensorRT
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu)
    find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/aarch64-linux-gnu)
    add_subdirectory(csrc/jetson)
endif()

# Summary
message(STATUS "")
message(STATUS "[mast3r] Build configuration:")
message(STATUS "  CPU backend:    ${BUILD_CPU}")
message(STATUS "  CUDA backend:   ${BUILD_CUDA}")
message(STATUS "  MPS backend:    ${BUILD_MPS}")
message(STATUS "  Jetson backend: ${BUILD_JETSON}")
message(STATUS "")