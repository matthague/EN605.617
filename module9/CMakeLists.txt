cmake_minimum_required(VERSION 3.21)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(assignment CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)

find_package(CUDAToolkit)

add_executable(thrustAssignment thrustAssignment.cu)

add_executable(nppNvgraphAssignment nppNvgraphAssignment.cu)

target_link_libraries(nppNvgraphAssignment PRIVATE CUDA::cudart CUDA::npps CUDA::nppisu CUDA::nppicc CUDA::nppicom CUDA::nppidei CUDA::nppif CUDA::nppig CUDA::nppim CUDA::nppist CUDA::nppitc CUDA::nppc CUDA::nppial nvgraph freeimage)

target_include_directories(nppNvgraphAssignment PRIVATE ${CMAKE_SOURCE_DIR}/common/UtilNPP ${CMAKE_SOURCE_DIR}/common/FreeImage/include)

set_target_properties(
        thrustAssignment
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        nppNvgraphAssignment
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)