cmake_minimum_required(VERSION 3.24)
project(nano_qwen35_moe_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ExecuTorch root — adjust if building from a different location
set(EXECUTORCH_ROOT $ENV{EXECUTORCH_ROOT})
if(NOT EXECUTORCH_ROOT)
  # Default: assume executorch is a sibling directory
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../executorch)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# gflags
set(gflags_DIR ${EXECUTORCH_ROOT}/cmake-out/third-party/gflags)
find_package(gflags REQUIRED)

# executorch
list(APPEND CMAKE_FIND_ROOT_PATH ${EXECUTORCH_ROOT}/cmake-out)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries executorch gflags)

# CPU ops
list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

# Extensions
list(APPEND link_libraries
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
  extension_named_data_map
)

# CUDA backend (optional)
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND link_libraries aoti_cuda_backend)
  if(NOT MSVC)
    executorch_target_link_options_shared_lib(aoti_cuda_backend)
  endif()
endif()

add_executable(runner runner.cpp)
target_include_directories(runner PUBLIC ${_common_include_directories})
target_link_libraries(runner PUBLIC ${link_libraries})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(runner PRIVATE "LINKER:-s")
  endif()
endif()
