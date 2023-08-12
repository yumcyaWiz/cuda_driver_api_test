cmake_minimum_required(VERSION 3.27)
project(cuda-driver-api-test LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ version and std
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# CUDA C++ version and std
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# kernel
add_library(kernel OBJECT "src/kernel.cu")
set_target_properties(kernel PROPERTIES
    CUDA_PTX_COMPILATION ON
)

# main
add_executable(main "src/main.cpp")
target_link_libraries(main PRIVATE
    CUDA::cuda_driver
)