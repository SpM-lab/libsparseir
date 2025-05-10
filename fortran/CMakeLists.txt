# Fortran bindings for SparseIR

# Enable Fortran preprocessor
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

file(GLOB TYPE_FILES
    "_*_proc.inc"
    "_*_impl.inc"
)

add_library(sparseir_fortran SHARED
    sparseir.F90
    _fortran_types.inc
    _impl_types.inc
    _fortran_assign.inc
    _cbinding.inc
    ${TYPE_FILES}
)

set_target_properties(sparseir_fortran PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(sparseir_fortran PRIVATE sparseir)

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
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sparseir/fortran
    FILES_MATCHING PATTERN "*.mod"
)

if(SPARSEIR_BUILD_TESTING)
    add_subdirectory(test)
endif()
