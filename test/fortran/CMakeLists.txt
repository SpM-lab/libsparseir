# Add test executables
add_executable(test_integration test_integration.f90)

# モジュールファイルのディレクトリを取得
get_target_property(SPARSEIR_FORTRAN_MODDIR sparseir_fortran Fortran_MODULE_DIRECTORY)

# includeディレクトリを明示的に追加
target_include_directories(test_integration PRIVATE
    ${SPARSEIR_FORTRAN_MODDIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../fortran
)

target_link_libraries(test_integration PRIVATE sparseir_fortran sparseir)

set_target_properties(test_integration PROPERTIES
    Fortran_MODULE_DIRECTORY ${SPARSEIR_FORTRAN_MODDIR}
)

# Add tests
add_test(NAME fortran_test_integration
    COMMAND test_integration
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
) 