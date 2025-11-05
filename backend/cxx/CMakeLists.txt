cmake_minimum_required(VERSION 3.10)

# Extract version from header
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/modules")
include("VersionFromHeader")
version_from_header(SPARSEIR_VERSION
    HEADER "include/sparseir/version.h"
    MACROS SPARSEIR_VERSION_MAJOR SPARSEIR_VERSION_MINOR SPARSEIR_VERSION_PATCH
)

project(SparseIR
    VERSION ${SPARSEIR_VERSION}
    LANGUAGES CXX
    DESCRIPTION "SparseIR C++ library"
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set, defaulting to Release")
endif()

# Options
option(SPARSEIR_USE_EXTERN_FBLAS_PTR "Use external BLAS function pointers (for Python bindings)" OFF)
option(SPARSEIR_USE_BLAS_ILP64 "Enable ILP64 BLAS interface (64-bit integers)" OFF)
option(SPARSEIR_BUILD_TESTING "Enable testing" OFF)
option(SPARSEIR_DEBUG "Enable debug logging" OFF)

if(SPARSEIR_DEBUG)
    add_definitions(-DDEBUG_SPIR)
endif()

# Find Eigen3
set(EIGEN3_REQUIRED_VERSION "3.4.1")
find_package(Eigen3 ${EIGEN3_REQUIRED_VERSION} QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching from GitLab...")
    include(FetchContent)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG ${EIGEN3_REQUIRED_VERSION}
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

# Find Threads
find_package(Threads REQUIRED)

# Fetch XPrec
include(FetchContent)
FetchContent_Declare(XPrec
    GIT_REPOSITORY https://github.com/tuwien-cms/libxprec
    GIT_TAG v0.7.0
)
FetchContent_GetProperties(XPrec)
if(NOT xprec_POPULATED)
    FetchContent_Populate(XPrec)
endif()

# BLAS configuration
if(SPARSEIR_USE_EXTERN_FBLAS_PTR)
    # External function pointer mode: BLAS is provided at runtime, no linking needed
    message(STATUS "BLAS will be provided via external function pointers at runtime")
    add_compile_definitions(SPARSEIR_USE_EXTERN_FBLAS_PTR)
    
    # Set compile-time ILP64/LP64 selection
    if(SPARSEIR_USE_BLAS_ILP64)
        message(STATUS "Using ILP64 interface (64-bit integers) for external BLAS")
        add_compile_definitions(SPARSEIR_USE_BLAS_ILP64)
    else()
        message(STATUS "Using LP64 interface (32-bit integers, default) for external BLAS")
    endif()
else()
    # Link-time BLAS mode: find and link BLAS library
    # Set BLAS integer size based on SPARSEIR_USE_BLAS_ILP64
    if(SPARSEIR_USE_BLAS_ILP64)
        set(BLA_SIZEOF_INTEGER 8)
        message(STATUS "Searching for ILP64 BLAS interface (64-bit integers)")
    else()
        set(BLA_SIZEOF_INTEGER 4)
        message(STATUS "Searching for LP64 BLAS interface (32-bit integers, default)")
    endif()
    
    # Find BLAS package using standard CMake FindBLAS
    find_package(BLAS REQUIRED)
    
    if(BLAS_FOUND)
        message(STATUS "BLAS libraries found: ${BLAS_LIBRARIES}")
        if(SPARSEIR_USE_BLAS_ILP64)
            message(STATUS "BLAS with ILP64 interface (BLA_SIZEOF_INTEGER=8)")
            add_compile_definitions(SPARSEIR_USE_BLAS_ILP64)
        else()
            message(STATUS "BLAS with LP64 interface (BLA_SIZEOF_INTEGER=4)")
        endif()
    else()
        message(FATAL_ERROR "BLAS libraries not found - BLAS is required")
    endif()
endif()

# C++ Library Build
add_library(sparseir SHARED
    src/utils.cpp
    src/linalg.cpp
    src/root.cpp
    src/specfuncs.cpp
    src/svd.cpp
    src/sve.cpp
    src/poly.cpp
    src/kernel.cpp
    src/gemm.cpp
    src/cinterface.cpp
)

if(NOT MSVC)
    target_compile_options(sparseir PRIVATE -Wall -Wextra -pedantic)
endif()

# Intel Compiler: disable fastmath optimizations that break xprec's extended precision
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(sparseir PRIVATE -fp-model=precise)
endif()

target_include_directories(sparseir
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        $<BUILD_INTERFACE:${xprec_SOURCE_DIR}/include>
)

set_target_properties(sparseir PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(sparseir PRIVATE Eigen3::Eigen Threads::Threads)

# BLAS/LAPACK linking (only when not using external function pointers)
if(NOT SPARSEIR_USE_EXTERN_FBLAS_PTR)
    if(BLAS_LIBRARIES OR BLAS_FOUND)
        target_link_libraries(sparseir PRIVATE ${BLAS_LIBRARIES})
        message(STATUS "Linked BLAS libraries: ${BLAS_LIBRARIES}")
    else()
        message(FATAL_ERROR "BLAS libraries not found - BLAS is required")
    endif()
else()
    message(STATUS "BLAS linking skipped (using external function pointers)")
endif()

# Library convention alias
add_library(SparseIR::sparseir ALIAS sparseir)

# Testing (if enabled)
if(SPARSEIR_BUILD_TESTING)
    enable_testing()
    
    # Fetch Catch2
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)
    
    add_subdirectory(test)
endif()

# Installation
install(TARGETS sparseir
    EXPORT sparseirTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT sparseir
)

# Install C API headers only
install(FILES
    include/sparseir/sparseir.h
    include/sparseir/version.h
    include/sparseir/spir_status.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sparseir"
    COMPONENT sparseir
)

# CMake package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Create SparseIRConfig.cmake
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/sparseirConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIR"
)

install(EXPORT sparseirTargets
    FILE SparseIRTargets.cmake
    NAMESPACE SparseIR::
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIR"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIR"
)
