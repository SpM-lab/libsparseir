# Fortran bindings for SparseIR

add_library(sparseir_fortran SHARED
    sparseir.f90
    sparseir_impl.f90
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
    DESTINATION "${SPARSEIR_INSTALL_LIBDIR}"
    COMPONENT sparseir
)

# Install Fortran module files
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sparseir/fortran
    FILES_MATCHING PATTERN "*.mod"
)

# Build Fortran test program
add_executable(test_kernel test_kernel.f90)
target_link_libraries(test_kernel PRIVATE sparseir_fortran)
set_target_properties(test_kernel PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran
)

# Add Fortran test to CTest if testing is enabled
if(SPARSEIR_BUILD_TESTING)
    add_test(
        NAME test_kernel
        COMMAND test_kernel
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif() 