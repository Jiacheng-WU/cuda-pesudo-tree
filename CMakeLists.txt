cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# create project
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Set CUDA architectures to 'native' if not already defined
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# add executable
add_executable(main main.cpp test.cu)

# add dependencies
set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/third_party/ CACHE PATH "CPM source cache")
include(cmake/CPM.cmake)

# CPMAddPackage("gh:fmtlib/fmt#12.0.0")
CPMAddPackage("gh:nlohmann/json@3.12.0")
CPMAddPackage("gh:catchorg/Catch2@3.10.0")

# link dependencies
target_link_libraries(main nlohmann_json::nlohmann_json Catch2::Catch2WithMain)
