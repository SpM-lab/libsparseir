cmake_minimum_required(VERSION 3.10)

# Include GNUInstallDirs for standard install paths
include(GNUInstallDirs)

# Find dependencies first to get version
find_package(SparseIR REQUIRED)

# Set project with version from SparseIR
project(SparseIR_Fortran 
    VERSION ${SparseIR_VERSION}
    LANGUAGES Fortran
)

# Enable Fortran preprocessor
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

# Set Fortran module directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran)

add_library(sparseir_fortran SHARED
    sparseir.f90
    sparseir_ext.F90
    _cbinding.inc
    _cbinding_public.inc
    ${TYPE_FILES}
)

# Add compiler-specific flags for heap arrays and fastmath
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    # Intel Fortran: Put all arrays on heap to avoid stack overflow
    #target_compile_options(sparseir_fortran PRIVATE -heap-arrays 0)
    # Intel C++ uses -fp-model=precise to disable fastmath for xprec compatibility
    # This is handled in the C++ backend CMakeLists.txt
endif()

# Add include directory
target_include_directories(sparseir_fortran PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Add interface include directory for installed module files
target_include_directories(sparseir_fortran INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran>
    $<INSTALL_INTERFACE:include/sparseir>
)

set_target_properties(sparseir_fortran PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH ON
)

target_link_libraries(sparseir_fortran PUBLIC SparseIR::sparseir)

# Add alias
add_library(SparseIR::sparseir_fortran ALIAS sparseir_fortran)

# Testing (if enabled via option)
option(SPARSEIR_BUILD_TESTING "Enable creation of SparseIR tests" OFF)
if(SPARSEIR_BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

# Install Fortran library
install(TARGETS sparseir_fortran
    EXPORT sparseirFortranTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT sparseir_fortran
)

# Install Fortran module files
install(
    DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sparseir"
    FILES_MATCHING PATTERN "*.mod"
)

# CMake package config for Fortran bindings
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRFortranConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT sparseirFortranTargets
    FILE SparseIRFortranTargets.cmake
    NAMESPACE SparseIR::
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIRFortran"
)
