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

# add dependencies
set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/third_party/ CACHE PATH "CPM source cache")
include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#12.0.0")
CPMAddPackage("gh:nlohmann/json@3.12.0")
CPMAddPackage("gh:abseil/abseil-cpp#20250814.1")
CPMAddPackage("gh:boost-ext/ut@2.3.1")
CPMAddPackage(
    URI "gh:microsoft/proxy#4.0.0"
    OPTIONS "BUILD_TESTING OFF"
)
CPMAddPackage("gh:pybind/pybind11@3.0.1")
CPMAddPackage(
    URI "gh:NVIDIA/cutlass@4.2.1"
    OPTIONS "CUTLASS_ENABLE_EXAMPLES OFF"
            "CUTLASS_ENABLE_TESTS OFF"
            "CUTLASS_ENABLE_TOOLS OFF"
)
# CPMAddPackage(
#     NAME stdexec
#     GIT_REPOSITORY "https://github.com/NVIDIA/stdexec.git"
#     GIT_TAG "main"
#     OPTIONS "STDEXEC_ENABLE_CUDA ON"
#             "STDEXEC_BUILD_TESTS OFF"
#             "STDEXEC_BUILD_EXAMPLES OFF"
# )

# Should Contain A Patch to Pass Compilation or Just Compile LP
# CPMAddPackage(
#     NAME cuopt
#     GIT_REPOSITORY "https://github.com/NVIDIA/cuopt.git"
#     GIT_TAG "branch-25.10"
#     SOURCE_SUBDIR "cpp"
#     OPTIONS "CMAKE_CXX_FLAGS -Wno-error=format"
#             "BUILD_EXAMPLES OFF"
#             "BUILD_TESTS OFF"
# )

# create global include directory
include_directories(include)

# create blob for all cu files
file(GLOB_RECURSE CUDA_SOURCE "src/*.cu")
# add executable
# add_library(myptx OBJECT ${CUDA_SOURCE})
# set_target_properties(myptx PROPERTIES
#     CUDA_PTX_COMPILATION ON
# )
add_library(mycudalib STATIC ${CUDA_SOURCE})
set_target_properties(mycudalib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(mycudalib
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>
    PRIVATE --extended-lambda)

target_link_libraries(mycudalib
    PUBLIC CUDA::cudart
    PUBLIC CUDA::cublas
    PUBLIC nvidia::cutlass::cutlass
)

file (GLOB_RECURSE CPP_SOURCE "src/*.cpp")
add_library(mycpplib STATIC ${CPP_SOURCE})
target_link_libraries(mycpplib
    PUBLIC fmt::fmt
    PUBLIC nlohmann_json::nlohmann_json
    PUBLIC absl::log
    PUBLIC msft_proxy4::proxy
    # PUBLIC cuopt::cuopt
)

add_executable(main main.cpp)
target_link_libraries(main
    PRIVATE mycudalib
    PRIVATE mycpplib)


pybind11_add_module(pesudo_tree python/bind/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(pesudo_tree
                           PRIVATE VERSION_INFO=1.0)

enable_testing()
add_executable(tests test/test.cpp)
target_link_libraries(tests
    PRIVATE mycudalib
    PRIVATE mycpplib
    PUBLIC Boost::ut)
add_test(NAME all_tests COMMAND tests)
