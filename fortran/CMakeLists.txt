# Fortran bindings for SparseIR

# Enable Fortran preprocessor
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

# Set Fortran module directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran)

file(GLOB TYPE_FILES
    "_*_proc.inc"
    "_*_impl.inc"
)

add_library(sparseir_fortran SHARED
    sparseir.f90
    sparseir_ext.F90
    _cbinding.inc
    _cbinding_public.inc
    ${TYPE_FILES}
)

# Add include directory
target_include_directories(sparseir_fortran PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/fortran
)

# Add interface include directory for installed module files
target_include_directories(sparseir_fortran INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/sparseir>
)

set_target_properties(sparseir_fortran PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(sparseir_fortran PUBLIC sparseir)

# Add alias
add_library(SparseIR::sparseir_fortran ALIAS sparseir_fortran)

# Install Fortran library
install(TARGETS sparseir_fortran
    EXPORT sparseirTargets
    LIBRARY DESTINATION "${SPARSEIR_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${SPARSEIR_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${SPARSEIR_INSTALL_BINDIR}"
    COMPONENT sparseir
)

# Install Fortran module files
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran/
    DESTINATION "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/sparseir"
    FILES_MATCHING PATTERN "*.mod"
)