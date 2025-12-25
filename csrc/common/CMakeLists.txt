# Common library shared by all backends

add_library(mast3r_common STATIC
    types.cpp
    model_loader.cpp
)

target_include_directories(mast3r_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Eigen for linear algebra (header-only)
find_package(Eigen3 3.4 QUIET)
if(Eigen3_FOUND)
    target_link_libraries(mast3r_common PUBLIC Eigen3::Eigen)
    target_compile_definitions(mast3r_common PUBLIC MAST3R_HAS_EIGEN)
    message(STATUS "[mast3r] Eigen3 found: ${Eigen3_VERSION}")
else()
    message(STATUS "[mast3r] Eigen3 not found, using bundled headers")
endif()