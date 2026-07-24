cmake_minimum_required(VERSION 3.18)

# Optional; cpp_setup passes -DENABLE_CUDA=ON / -DENABLE_HIP=ON when needed
if(NOT DEFINED ENABLE_CUDA)
    set(ENABLE_CUDA OFF)
endif()
if(NOT DEFINED ENABLE_HIP)
    set(ENABLE_HIP OFF)
endif()
if(NOT DEFINED PK_DEBUG_BUILD)
    set(PK_DEBUG_BUILD OFF)
endif()

# Setup
if(ENABLE_CUDA)
    project(PyKokkosModule CXX CUDA)
elseif(ENABLE_HIP)
    project(PyKokkosModule CXX HIP)
else()
    project(PyKokkosModule CXX)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies
find_package(pybind11 2.11 CONFIG REQUIRED)
find_package(Kokkos 4.7.1 EXACT REQUIRED)

# Sources and targets
file(GLOB SOURCE_FILES "*.cpp")
add_library(${MODULE_NAME} SHARED ${SOURCE_FILES})

set_target_properties(${MODULE_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Includes
target_include_directories(${MODULE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Compiler definitions and flags
target_compile_definitions(${MODULE_NAME} PRIVATE
    SPACE=${PK_EXEC_SPACE}
    pk_arg_memspace=${PK_ARG_MEMSPACE}
    pk_arg_layout=${PK_ARG_LAYOUT}
    pk_exec_space=Kokkos::${PK_EXEC_SPACE}
    pk_real=${PK_REAL}
)

# Debug builds (Python ExecutionSpace.DebugCuda / DebugHIP / DebugOpenMP)
if(PK_DEBUG_BUILD)
    if(ENABLE_CUDA)
        # nvcc_wrapper / nvcc: device debug (-G), symbols (-g), -O0; -keep keeps intermediates
        target_compile_options(${MODULE_NAME} PRIVATE -G -g -O0 -keep)
    elseif(ENABLE_HIP)
        target_compile_options(${MODULE_NAME} PRIVATE -g -O0)
    else()
        # OpenMP / Serial / host parallel
        target_compile_options(${MODULE_NAME} PRIVATE -g -O0)
    endif()
endif()

# Link libraries
target_link_libraries(${MODULE_NAME} PRIVATE pybind11::module Kokkos::kokkos)
target_link_libraries(${MODULE_NAME} PRIVATE Kokkos::kokkoscore${LIB_SUFFIX} Kokkos::kokkoscontainers${LIB_SUFFIX})

# Install
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION .)
