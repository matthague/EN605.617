cmake_minimum_required(VERSION 3.8)
project(assignment CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)

FindCUDA()

add_executable(assignment assignment.cu)


target_link_libraries(assignment PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)

set_target_properties(
        assignment
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
