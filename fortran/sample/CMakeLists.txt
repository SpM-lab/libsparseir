cmake_minimum_required(VERSION 3.10)
project(libsparseir-fortran-sample LANGUAGES Fortran C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Fortran standard
set(CMAKE_Fortran_STANDARD 2008)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_TESTING "Build tests" OFF)
option(USE_SYSTEM_LIBSPARSEIR "Use system-installed libsparseir" OFF)

# Setup libsparseir dependency
if(USE_SYSTEM_LIBSPARSEIR)
    find_package(SparseIR REQUIRED)
    set(SPARSEIR_TARGET SparseIR::sparseir_fortran)
else()
    # Fetch and build libsparseir from source
    include(FetchContent)

    FetchContent_Declare(
        libsparseir
        GIT_REPOSITORY https://github.com/SpM-lab/libsparseir.git
        GIT_TAG main
    )

    # Set options for libsparseir build
    set(SPARSEIR_BUILD_FORTRAN ON CACHE BOOL "Build Fortran bindings for libsparseir")
    set(SPARSEIR_BUILD_TESTING OFF CACHE BOOL "Disable libsparseir tests")

    FetchContent_MakeAvailable(libsparseir)

    set(SPARSEIR_TARGET SparseIR::sparseir_fortran)
endif()

find_library(FFTW3_LIB fftw3)
find_path(FFTW3_INCLUDE_DIR fftw3.f)

# Create a simple Fortran test
add_executable(second_order_perturbation_fort ${CMAKE_CURRENT_SOURCE_DIR}/second_order_perturbation_fort.f90)

# Link with libsparseir
target_link_libraries(second_order_perturbation_fort PRIVATE ${SPARSEIR_TARGET} ${FFTW3_LIB})
target_include_directories(second_order_perturbation_fort PRIVATE 
    $<TARGET_PROPERTY:${SPARSEIR_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
    ${FFTW3_INCLUDE_DIR}
)