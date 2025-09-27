cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# create project
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Make Compile Commands available for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

# Set CUDA architectures to 'native' if not already defined
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost CONFIG)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
# Use most recent fmt in CPM from source
# find_package(fmt CONFIG REQUIRED)
find_package(absl REQUIRED COMPONENTS log)
find_package(nlohmann_json CONFIG REQUIRED)

set(PYBIND11_FINDPYTHON NEW)
find_package(pybind11 CONFIG REQUIRED)

# Follow CUTLASS cmake INTERFACE pattern
find_path(CUTLASS_INCLUDE_DIRS "cutlass/cutlass.h"
          HINTS ENV CMAKE_PREFIX_PATH
          PATH_SUFFIXES "include")
add_library(CUTLASS INTERFACE)
add_library(nvidia::cutlass::cutlass ALIAS CUTLASS)
set_target_properties(CUTLASS PROPERTIES EXPORT_NAME cutlass)
target_include_directories(CUTLASS INTERFACE
  ${CUTLASS_INCLUDE_DIRS}
)

# add dependencies
set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/third_party/ CACHE PATH "CPM source cache")
include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#12.0.0")
CPMAddPackage(
    URI "gh:microsoft/proxy#4.0.0"
    OPTIONS "BUILD_TESTING OFF"
)


# create global include directory
include_directories(include)

# create blob for all cu files
file(GLOB_RECURSE CUDA_SOURCE "src/*.cu")
add_library(mycudalib STATIC ${CUDA_SOURCE})
set_target_properties(mycudalib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Though CUDA_PTX_COMPILATION can be used to generate PTX files, it is not
# suitable here because we want to compile to binary for the native architecture.
# Thus use --keep to keep the intermediate files which include the ptx files.
target_compile_options(mycudalib
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>
    PRIVATE --extended-lambda)
target_link_libraries(mycudalib
    PUBLIC CUDA::cudart
    PUBLIC CUDA::cublas
    PRIVATE nvidia::cutlass::cutlass
)

# create blob for all cpp files
file (GLOB_RECURSE CPP_SOURCE "src/*.cpp")
add_library(mycpplib STATIC ${CPP_SOURCE})
target_link_libraries(mycpplib
    PUBLIC fmt::fmt
    PUBLIC nlohmann_json::nlohmann_json
    PUBLIC absl::log
    PUBLIC msft_proxy4::proxy
    PUBLIC Boost::boost
    PUBLIC Threads::Threads
)

add_executable(main main.cpp)
target_link_libraries(main
    PRIVATE mycudalib
    PRIVATE mycpplib)

# For Python bindings
pybind11_add_module(pesudo_tree python/bind/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(pesudo_tree
                           PRIVATE VERSION_INFO=1.0)


# Do Testing
enable_testing()
add_executable(tests test/test.cpp)
target_link_libraries(tests
    PRIVATE mycudalib
    PRIVATE mycpplib
    PRIVATE GTest::gtest_main)
add_test(NAME all_tests COMMAND tests)
