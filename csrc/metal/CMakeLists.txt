# Metal Backend - Apple Silicon native

# Compile Metal shaders
set(METAL_SHADERS
    preprocessing
    attention
    matching
    topk_sort
    vit
    dpt_heads
)

set(METAL_AIR_FILES)
foreach(shader ${METAL_SHADERS})
    set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.metal)
    set(SHADER_AIR ${CMAKE_CURRENT_BINARY_DIR}/${shader}.air)

    add_custom_command(
        OUTPUT ${SHADER_AIR}
        COMMAND xcrun -sdk macosx metal -c ${SHADER_SRC} -o ${SHADER_AIR}
        DEPENDS ${SHADER_SRC}
        COMMENT "Compiling ${shader}.metal"
    )
    list(APPEND METAL_AIR_FILES ${SHADER_AIR})
endforeach()

# Combine all .air files into metallib
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mast3r.metallib
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${CMAKE_CURRENT_BINARY_DIR}/mast3r.metallib
    DEPENDS ${METAL_AIR_FILES}
    COMMENT "Linking Metal shaders into metallib"
)

add_custom_target(metal_shaders DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mast3r.metallib)

# Metal engine library
add_library(mast3r_metal_impl STATIC
    metal_context.mm
    metal_engine.mm
    vit_buffers.mm
)

target_include_directories(mast3r_metal_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
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

# Install to package
install(TARGETS _metal LIBRARY DESTINATION mast3r_runtime/backends)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mast3r.metallib DESTINATION mast3r_runtime/backends)