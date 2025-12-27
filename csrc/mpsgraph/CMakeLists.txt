# MPSGraph Backend - Apple Silicon with native SDPA
# Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
# Requires macOS 15+ for scaledDotProductAttention

find_library(MPS_LIBRARY MetalPerformanceShaders REQUIRED)
find_library(MPSGRAPH_LIBRARY MetalPerformanceShadersGraph REQUIRED)

# MPSGraph engine library
add_library(mast3r_mpsgraph_impl STATIC
    mpsgraph_context.mm
    mpsgraph_engine.mm
)

target_include_directories(mast3r_mpsgraph_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(mast3r_mpsgraph_impl PUBLIC
    mast3r_common
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${ACCELERATE_LIBRARY}
    ${MPS_LIBRARY}
    ${MPSGRAPH_LIBRARY}
)

target_compile_options(mast3r_mpsgraph_impl PRIVATE
    -fobjc-arc
)

# Require macOS 15 for SDPA
target_compile_definitions(mast3r_mpsgraph_impl PRIVATE
    MAST3R_MPSGRAPH_SDPA=1
)

# Python bindings
pybind11_add_module(_mpsgraph bindings.mm)
target_link_libraries(_mpsgraph PRIVATE mast3r_mpsgraph_impl)
target_compile_options(_mpsgraph PRIVATE -fobjc-arc)

# Install to package
install(TARGETS _mpsgraph LIBRARY DESTINATION mast3r_runtime/backends)
