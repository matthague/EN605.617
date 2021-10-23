cmake_minimum_required(VERSION 3.21)
project(assignment CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)

find_package(CUDAToolkit)

add_executable(assignment assignment.cu)

target_link_libraries(assignment PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft)

set_target_properties(
        assignment
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
