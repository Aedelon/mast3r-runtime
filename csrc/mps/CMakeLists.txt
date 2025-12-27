# MPS Backend - Apple Silicon with MPSGraph SDPA
# Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
# Requires macOS 15+ for scaledDotProductAttention

find_library(MPS_LIBRARY MetalPerformanceShaders REQUIRED)
find_library(MPSGRAPH_LIBRARY MetalPerformanceShadersGraph REQUIRED)

# MPS engine library
add_library(mast3r_mps_impl STATIC
    mpsgraph_context.mm
    mpsgraph_engine.mm
    gpu_tensor.mm
)

target_include_directories(mast3r_mps_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(mast3r_mps_impl PUBLIC
    mast3r_common
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${ACCELERATE_LIBRARY}
    ${MPS_LIBRARY}
    ${MPSGRAPH_LIBRARY}
)

target_compile_options(mast3r_mps_impl PRIVATE
    -fobjc-arc
)

# Require macOS 15 for SDPA
target_compile_definitions(mast3r_mps_impl PRIVATE
    MAST3R_MPS_SDPA=1
)

# Python bindings
pybind11_add_module(_mps bindings.mm)
target_link_libraries(_mps PRIVATE mast3r_mps_impl)
target_compile_options(_mps PRIVATE -fobjc-arc)

# Install to package
install(TARGETS _mps LIBRARY DESTINATION mast3r_runtime/backends)