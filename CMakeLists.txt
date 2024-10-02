cmake_minimum_required (VERSION 3.9)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)

project(libsparseir C CXX)

# Eigen3
find_package (Eigen3 3.4 REQUIRED NO_MODULE)

# libxprec
include(FetchContent)
FetchContent_Declare(XPrec
    GIT_REPOSITORY https://github.com/tuwien-cms/libxprec
    GIT_TAG v0.5.0
    )
FetchContent_MakeAvailable(XPrec)

add_executable(main src/main.cpp )
target_link_libraries( main Eigen3::Eigen XPrec::xprec)

# -------------------------------------
# Testing

option(XPREC_BUILD_TESTING "Enable creation of xprec tests." ${BUILD_TESTING})

if (XPREC_BUILD_TESTING)
    enable_testing()
    add_subdirectory("test")
endif()