cmake_minimum_required(VERSION 3.21)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(assignment CUDA)

set(CMAKE_CUDA_STANDARD 23)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")

add_executable(assignment assignment.cu)

set_target_properties(
        assignment
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)