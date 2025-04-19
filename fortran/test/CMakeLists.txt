# Find Python
find_package(Python3 REQUIRED)

# Generate test file from README
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/readme_samples_test.f90
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_fortran_test.py
            ${CMAKE_SOURCE_DIR}/README.md
            ${CMAKE_CURRENT_BINARY_DIR}/readme_samples_test.f90
    DEPENDS ${CMAKE_SOURCE_DIR}/README.md
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_fortran_test.py
    COMMENT "Generating Fortran test file from README.md"
)

# Add test executables
add_executable(test_kernel test_kernel.f90)
target_link_libraries(test_kernel PRIVATE sparseir_fortran)
set_target_properties(test_kernel PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../fortran
)

add_executable(readme_samples_fortran_test
    ${CMAKE_CURRENT_BINARY_DIR}/readme_samples_test.f90
)
target_link_libraries(readme_samples_fortran_test
    PRIVATE sparseir_fortran
)
set_target_properties(readme_samples_fortran_test PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../fortran
)

# Add tests
add_test(NAME test_kernel
    COMMAND test_kernel
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(NAME readme_samples_fortran_test
    COMMAND readme_samples_fortran_test
) 