cmake_minimum_required(VERSION 3.15)
project(SparseIR_CAPI_Tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(SparseIR REQUIRED)

# Find Eigen3
find_package(Eigen3 3.4.0 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found in system, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

# Find Threads
find_package(Threads REQUIRED)

# Find or fetch Catch2
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# Fetch XPrec for test utilities
FetchContent_Declare(XPrec
    GIT_REPOSITORY https://github.com/tuwien-cms/libxprec
    GIT_TAG v0.7.0
)
FetchContent_GetProperties(XPrec)
if(NOT xprec_POPULATED)
    FetchContent_Populate(XPrec)
endif()

# C-API integration test
add_executable(cinterface_integration cinterface_integration.cxx)

target_include_directories(cinterface_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${xprec_SOURCE_DIR}/include
)

target_link_libraries(cinterface_integration PRIVATE 
    Catch2::Catch2WithMain
    SparseIR::sparseir
    Eigen3::Eigen 
    Threads::Threads
)

# Intel Compiler: disable fastmath optimizations that break xprec's extended precision
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(cinterface_integration PRIVATE -fp-model=precise)
endif()

# Optional BLAS linking
find_package(BLAS QUIET)
if(BLAS_FOUND)
    target_link_libraries(cinterface_integration PRIVATE ${BLAS_LIBRARIES})
endif()

# macOS Accelerate framework
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(cinterface_integration PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
endif()

# Enable testing
enable_testing()
add_test(NAME cinterface_integration COMMAND cinterface_integration -d yes)
