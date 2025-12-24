# CPU Backend - Pure C++ with OpenMP/BLAS

# Find OpenMP
find_package(OpenMP)

# Find BLAS (OpenBLAS, MKL, or Accelerate on macOS)
if(APPLE)
    # Use Accelerate framework on macOS
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
    set(BLAS_LIBRARIES ${ACCELERATE_LIBRARY})
    set(BLAS_FOUND TRUE)
else()
    find_package(BLAS)
endif()

# CPU engine library
add_library(mast3r_cpu_impl STATIC
    cpu_engine.cpp
    preprocessing.cpp
    attention.cpp
    matching.cpp
)

target_include_directories(mast3r_cpu_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(mast3r_cpu_impl PUBLIC
    mast3r_common
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(mast3r_cpu_impl PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(mast3r_cpu_impl PUBLIC MAST3R_HAS_OPENMP)
endif()

if(BLAS_FOUND)
    target_link_libraries(mast3r_cpu_impl PUBLIC ${BLAS_LIBRARIES})
    target_compile_definitions(mast3r_cpu_impl PUBLIC MAST3R_HAS_BLAS)
endif()

# Python bindings
pybind11_add_module(_cpu bindings.cpp)
target_link_libraries(_cpu PRIVATE mast3r_cpu_impl)

# Install to package
install(TARGETS _cpu DESTINATION mast3r_runtime/backends)