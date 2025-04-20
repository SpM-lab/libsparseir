cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified, defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# extract version information from header file
if (NOT SPARSEIR_VERSION)
    include("VersionFromHeader")
    version_from_header(SPARSEIR_VERSION
        HEADER "include/sparseir/version.h"
        MACROS SPARSEIR_VERSION_MAJOR SPARSEIR_VERSION_MINOR SPARSEIR_VERSION_PATCH
        )
    message(STATUS "Extracted package version: ${SPARSEIR_VERSION}")

    set(SPARSEIR_VERSION "${SPARSEIR_VERSION}"
        CACHE STRING "version of the sparseir package")
    mark_as_advanced(SPARSEIR_VERSION)
endif()

# Configure project
project(
    SparseIR
    LANGUAGES CXX
    VERSION "${SPARSEIR_VERSION}"
    DESCRIPTION "SparseIR library"
    )

# Add Fortran language support if building Fortran bindings
option(SPARSEIR_BUILD_FORTRAN "Build Fortran bindings" OFF)
if(SPARSEIR_BUILD_FORTRAN)
    enable_language(Fortran)
    message(STATUS "Fortran support enabled for building Fortran bindings")
endif()

# Eigen3
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# libxprec - built only as a static dependency, not for installation
include(FetchContent)
FetchContent_Declare(XPrec
    GIT_REPOSITORY https://github.com/tuwien-cms/libxprec
    GIT_TAG mainline
)

# Extract the content but don't process it through add_subdirectory
FetchContent_GetProperties(XPrec)
if(NOT xprec_POPULATED)
    FetchContent_MakeAvailable(XPrec)
    message(STATUS "XPrec source directory: ${xprec_SOURCE_DIR}")
endif()

# Add XPrec include directory
include_directories(${xprec_SOURCE_DIR}/include)

# ---------------------------------
# Building

# Enable debug logging
#add_definitions(-DDEBUG_CINTERFACE)

# Set the default component name for installations
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME sparseir)

add_library(sparseir SHARED
    src/utils.cpp
    src/linalg.cpp
    src/root.cpp
    src/specfuncs.cpp
    src/svd.cpp
    src/sve.cpp
    src/poly.cpp
    src/kernel.cpp
    src/cinterface.cpp
    )
if(NOT MSVC)
    target_compile_options(sparseir PRIVATE -Wall -Wextra -pedantic)
endif()

target_include_directories(sparseir PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${XPrec_SOURCE_DIR}/include>
    )
set_target_properties(sparseir PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    )

target_link_libraries(sparseir PRIVATE Eigen3::Eigen Threads::Threads)

# Use library convention.
add_library(SparseIR::sparseir ALIAS sparseir)

# Build Fortran bindings if enabled
if(SPARSEIR_BUILD_FORTRAN)
    add_subdirectory(fortran)
endif()

# Add include paths from parent scope if available

# -------------------------------------
# Testing

option(SPARSEIR_BUILD_TESTING "Enable creation of SparseIR tests." ${BUILD_TESTING})

if (SPARSEIR_BUILD_TESTING)
    enable_testing()

    # Download and configure Catch2 for testing
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    # Pass include paths to test subdirectory
    add_subdirectory("test")
    add_subdirectory("testsample")
endif()

# -------------------------------------
# Installation

# Provides CMAKE_INSTALL_* directories
include(GNUInstallDirs)
set(SPARSEIR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE PATH "directory into which to install sparseir include files")
set(SPARSEIR_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "directory into which to install sparseir library files")

# Install only the C API library and headers
install(TARGETS sparseir
    DESTINATION "${SPARSEIR_INSTALL_LIBDIR}"
    COMPONENT sparseir
)

# Install only necessary header files (C API)
install(FILES
    include/sparseir/sparseir.h
    include/sparseir/version.h
    DESTINATION "${SPARSEIR_INSTALL_INCLUDEDIR}/sparseir"
    COMPONENT sparseir
)