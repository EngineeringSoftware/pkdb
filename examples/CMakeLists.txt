# THIS IS SAMPLE CMAKE LIST, DO NOT USE IT IN ACTUAL PROJECT

# CMakeList that allows to run Kokkos applications with `cuda-gdb` and inster
# breakpoints inside Kokkos kernels, a.k.a. `operator()`

# Main feature is `-G -g -O0` flags in `target_compile_options`, that should be
# used in the actual cmake list

cmake_minimum_required(VERSION 3.16)
project(pp LANGUAGES CXX CUDA)

# Host and device debug flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/4.7.01.zip
)
FetchContent_MakeAvailable(Kokkos)

add_executable(main main.cpp)
target_link_libraries(main Kokkos::kokkos)

# Add -G for CUDA device debugging (nvcc_wrapper recognizes this)
target_compile_options(main PRIVATE -G -g -O0)
