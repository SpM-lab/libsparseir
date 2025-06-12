# Define test files
set(TEST_FILES
    test_integration
    test_ext
    second_order_perturbation_fort
)

find_library(FFTW3_LIB fftw3)
find_path(FFTW3_INCLUDE_DIR fftw3.f)

# Add test executables
foreach(TEST ${TEST_FILES})
    add_executable(${TEST} ${TEST}.f90)
endforeach()

get_target_property(SPARSEIR_FORTRAN_MODDIR sparseir_fortran Fortran_MODULE_DIRECTORY)

# Process each test
foreach(TEST ${TEST_FILES})
    target_include_directories(${TEST} PRIVATE
        ${SPARSEIR_FORTRAN_MODDIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../fortran
        ${FFTW3_INCLUDE_DIR}
    )

    target_link_libraries(${TEST} PRIVATE sparseir_fortran sparseir ${FFTW3_LIB})

    set_target_properties(${TEST} PROPERTIES
        Fortran_MODULE_DIRECTORY ${SPARSEIR_FORTRAN_MODDIR}
    )

    # Add test
    if(${TEST} STREQUAL "second_order_perturbation_fort")
        add_test(NAME fortran_${TEST}
            COMMAND sh -c "./${TEST} < input.in"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    else()
        add_test(NAME fortran_${TEST}
            COMMAND ${TEST}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()

    if(${TEST} STREQUAL "second_order_perturbation_fort")
    add_custom_command(TARGET ${TEST} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/input.in
            ${CMAKE_CURRENT_BINARY_DIR}/input.in
        )
    endif()
endforeach()