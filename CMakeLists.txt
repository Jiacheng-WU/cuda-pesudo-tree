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

# Append flag for Debug/RelWithDebInfo builds
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} --device-debug")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")

find_package(CUDAToolkit REQUIRED)
find_package(CCCL CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost CONFIG)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
# Use most recent fmt in CPM from source
# find_package(fmt CONFIG REQUIRED)
find_package(absl REQUIRED COMPONENTS log)
find_package(nlohmann_json CONFIG REQUIRED)


find_package(Python COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON NEW)
find_package(pybind11 CONFIG REQUIRED)

# Define variables for clarity

set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third_party/")
set(TORCH_DIR "${THIRD_PARTY_DIR}/libtorch")

# Only proceed if the extraction directory doesn't already exist
if(NOT EXISTS "${TORCH_DIR}")
    set(TORCH_URL "https://download.pytorch.org/libtorch/nightly/cu130/libtorch-shared-with-deps-latest.zip")
    set(TORCH_FILE "${CMAKE_BINARY_DIR}/libtorch.zip")
    message(STATUS "Downloading and extracting dependency to ${TORCH_DIR}")
    # 1. Download the ZIP file to a temporary location in the build directory
    file(DOWNLOAD ${TORCH_URL} ${TORCH_FILE} SHOW_PROGRESS)
    # 2. Decompress the ZIP directly into your desired source folder
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${TORCH_FILE}
        WORKING_DIRECTORY ${THIRD_PARTY_DIR}
        COMMAND_ERROR_IS_FATAL ANY
    )
else()
    message(STATUS "Dependency already exists at ${TORCH_DIR}")
endif()
set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_LIBRARY_ROOT})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${TORCH_DIR}/libtorch/share/cmake/Torch)

find_package(Torch CONFIG)

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
set(CPM_SOURCE_CACHE ${THIRD_PARTY_DIR} CACHE PATH "CPM source cache")
include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#12.0.0")
CPMAddPackage(
    URI "gh:microsoft/proxy#4.0.0"
    OPTIONS "BUILD_TESTING OFF"
)

# should include CCCL before stdexec
# otherwise stdexec may fail to use CUDA
CPMAddPackage(
    URI "gh:NVIDIA/stdexec#main"
    OPTIONS "STDEXEC_BUILD_TESTS OFF"
            "STDEXEC_BUILD_EXAMPLES OFF"
            "STDEXEC_BUILD_DOCS OFF"
            "STDEXEC_ENABLE_CUDA ON"
)

# create global include directory
include_directories(include)

add_library(core_utils INTERFACE)

target_link_libraries(core_utils
  INTERFACE
    fmt::fmt
    absl::log
    Boost::boost
)

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
    # Try to use C++23 standard, but nvcc does not support it
    # PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=c++23>
    PRIVATE --extended-lambda)
target_link_libraries(mycudalib
    PUBLIC CUDA::cudart
    PUBLIC CUDA::cublas
    PUBLIC nvidia::cutlass::cutlass
    PUBLIC ${TORCH_LIBRARIES}
    PRIVATE core_utils
)

# create blob for all cpp files
file (GLOB_RECURSE CPP_SOURCE "src/*.cpp")
add_library(mycpplib STATIC ${CPP_SOURCE})
target_link_libraries(mycpplib
    PUBLIC nlohmann_json::nlohmann_json
    PUBLIC msft_proxy4::proxy
    PUBLIC Threads::Threads
    PUBLIC STDEXEC::stdexec
    PUBLIC ${TORCH_LIBRARIES}
    PRIVATE core_utils
)

add_executable(main main.cpp)
target_link_libraries(main
    PRIVATE mycudalib
    PRIVATE mycpplib
    PRIVATE core_utils)

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
