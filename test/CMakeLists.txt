include(FetchContent)

set(CMAKE_CXX_STANDARD 11)

# Require Catch2 for tests
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
    FIND_PACKAGE_ARGS 3
    )

# Goes through find_package first
FetchContent_MakeAvailable(Catch2)

# Enable parallel testing with CTest
include(CTest)
enable_testing()

# List of all test files
set(TEST_FILES
    utils.cxx
    example.cxx
    _linalg.cxx
    _specfuncs.cxx
    _root.cxx
    gauss.cxx
    freq.cxx
    poly.cxx
    kernel.cxx
    svd.cxx
    sve.cxx
    basis.cxx
    augment.cxx
    sampling.cxx
    dlr.cxx
    cinterface.cxx
)

# Create individual test executables for each test file
foreach(TEST_FILE ${TEST_FILES})
    # Extract the base name without extension
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    
    # Create executable for this test file
    add_executable(test_${TEST_NAME} ${TEST_FILE})
    
    # Link against required libraries
    target_link_libraries(test_${TEST_NAME} PRIVATE Catch2::Catch2WithMain)
    target_link_libraries(test_${TEST_NAME} PRIVATE SparseIR::sparseir)
    
    # Add as a test to CTest
    add_test(NAME ${TEST_NAME} COMMAND test_${TEST_NAME})
endforeach()

# Keep the original combined test executable for backward compatibility
add_executable(libsparseirtests ${TEST_FILES})
target_link_libraries(libsparseirtests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(libsparseirtests PRIVATE SparseIR::sparseir)
add_test(NAME combined_tests COMMAND libsparseirtests)

#if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
    #set_property(TARGET tests PROPERTY CXX_STANDARD 20)
#else()
    #set_property(TARGET tests PROPERTY CXX_STANDARD 17)
#endif()
add_test(tests libsparseirtests)
