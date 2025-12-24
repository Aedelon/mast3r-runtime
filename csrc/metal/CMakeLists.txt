# Metal Backend - Apple Silicon native

# Compile Metal shaders
set(METAL_SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/preprocessing.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/attention.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matching.metal
)

# Create metallib from shaders
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mast3r.metallib
    COMMAND xcrun -sdk macosx metal -c ${METAL_SHADERS} -o ${CMAKE_CURRENT_BINARY_DIR}/mast3r.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/mast3r.air -o ${CMAKE_CURRENT_BINARY_DIR}/mast3r.metallib
    DEPENDS ${METAL_SHADERS}
    COMMENT "Compiling Metal shaders"
)

add_custom_target(metal_shaders DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mast3r.metallib)

# Metal engine library
add_library(mast3r_metal_impl STATIC
    metal_context.mm
    metal_engine.mm
)

target_include_directories(mast3r_metal_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(mast3r_metal_impl PUBLIC
    mast3r_common
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${ACCELERATE_LIBRARY}
)

target_compile_options(mast3r_metal_impl PRIVATE
    -fobjc-arc
)

add_dependencies(mast3r_metal_impl metal_shaders)

# Embed metallib path
target_compile_definitions(mast3r_metal_impl PRIVATE
    MAST3R_METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/mast3r.metallib"
)

# Python bindings
pybind11_add_module(_metal bindings.mm)
target_link_libraries(_metal PRIVATE mast3r_metal_impl)
target_compile_options(_metal PRIVATE -fobjc-arc)

# Install
install(TARGETS _metal DESTINATION mast3r_runtime/backends)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mast3r.metallib DESTINATION mast3r_runtime/backends)