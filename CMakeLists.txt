cmake_minimum_required(VERSION 3.10)

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

# Eigen3
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# libxprec
include(FetchContent)
FetchContent_Declare(XPrec
    GIT_REPOSITORY https://github.com/tuwien-cms/libxprec
    GIT_TAG mainline
    )
FetchContent_MakeAvailable(XPrec)

# ---------------------------------
# Building

add_library(sparseir SHARED
    src/cinterface.cpp
    )
if(NOT MSVC)
    target_compile_options(sparseir PRIVATE -Wall -Wextra -pedantic)
endif()

target_include_directories(sparseir PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    )
set_target_properties(sparseir PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    )

target_link_libraries(sparseir Eigen3::Eigen XPrec::xprec Threads::Threads)

# Use library convention.
add_library(SparseIR::sparseir ALIAS sparseir)
export(TARGETS sparseir
    NAMESPACE SparseIR::
    FILE SparseIRTargets.cmake)

# ------------------------------
# Installation

# Provides CMAKE_INSTALL_* directories
include(GNUInstallDirs)
set(SPARSEIR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE PATH "directory into which to install xprec include files")
set(SPARSEIR_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "directory into which to install xprec library files")
set(SPARSEIR_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/cmake/SparseIR"
    CACHE PATH "directory into which to install xprec cmake files")
option(SPARSEIR_INSTALL_CMAKE_PACKAGE "Installs CMake configuration files" ON)

install(TARGETS xprec
    DESTINATION "${SPARSEIR_INSTALL_LIBDIR}"
    EXPORT SparseIRTargets
    )
    install(DIRECTORY include/
    DESTINATION "${SPARSEIR_INSTALL_INCLUDEDIR}"
    PATTERN "ddouble-header-only.h" EXCLUDE
    )

if (SPARSEIR_INSTALL_CMAKE_PACKAGE)
    # These are essentially magic incantantions copied from the CMake tutorial:
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package/SparseIRConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfig.cmake"
        INSTALL_DESTINATION "${SPARSEIR_INSTALL_CMAKEDIR}"
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
        )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfigVersion.cmake"
        VERSION "${SPARSEIR_VERSION}"
        COMPATIBILITY SameMajorVersion
        )
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfigVersion.cmake"
        DESTINATION "${SPARSEIR_INSTALL_CMAKEDIR}"
        )
    install(EXPORT SparseIRTargets
        NAMESPACE SparseIR::
        DESTINATION "${SPARSEIR_INSTALL_CMAKEDIR}"
        )
endif()


# -------------------------------------
# Testing

option(SPARSEIR_BUILD_TESTING "Enable creation of xprec tests." ${BUILD_TESTING})

if (SPARSEIR_BUILD_TESTING)
    enable_testing()
    add_subdirectory("test")
endif()
