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

# Create a single combined test executable
add_executable(libsparseirtests ${TEST_FILES})
target_link_libraries(libsparseirtests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(libsparseirtests PRIVATE SparseIR::sparseir)
add_test(NAME combined_tests COMMAND libsparseirtests)

add_test(tests libsparseirtests)
