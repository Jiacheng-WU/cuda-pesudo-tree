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

find_package(CUDAToolkit REQUIRED)

# add dependencies
set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/third_party/ CACHE PATH "CPM source cache")
include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#12.0.0")
CPMAddPackage("gh:nlohmann/json@3.12.0")
CPMAddPackage("gh:catchorg/Catch2@3.10.0")

# create global include directory
include_directories(include)

# create blob for all cu files
file(GLOB_RECURSE CUDA_SOURCE "src/*.cu")
# add executable
add_library(myptx OBJECT ${CUDA_SOURCE})
set_target_properties(myptx PROPERTIES
    CUDA_PTX_COMPILATION ON
)
add_library(mycudalib STATIC ${CUDA_SOURCE})
set_target_properties(mycudalib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(mycudalib
    PUBLIC CUDA::cudart
    PUBLIC CUDA::cublas
)

file (GLOB_RECURSE CPP_SOURCE "src/*.cpp")
add_library(mycpplib STATIC ${CPP_SOURCE})
target_link_libraries(mycpplib
    PUBLIC fmt::fmt
    PUBLIC nlohmann_json::nlohmann_json
    PUBLIC Catch2::Catch2WithMain
)

add_executable(main main.cpp)
target_link_libraries(main
    PRIVATE mycudalib
    PRIVATE mycpplib)
