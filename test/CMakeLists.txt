include(FetchContent)

set(CMAKE_CXX_STANDARD 14)

# Use Catch2 from parent scope if available
if(NOT TARGET Catch2::Catch2WithMain)
    # Require Catch2 for tests
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
        )

    # Goes through find_package first
    FetchContent_MakeAvailable(Catch2)
endif()

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
target_link_libraries(libsparseirtests PRIVATE XPrec::xprec)

# Add include paths from parent scope if available
if(DEFINED TEST_INCLUDE_DIRS)
    target_include_directories(libsparseirtests PRIVATE ${TEST_INCLUDE_DIRS})
endif()

# Also explicitly include Eigen3 and XPrec include directories
if(DEFINED EIGEN3_INCLUDE_DIR)
    target_include_directories(libsparseirtests PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

if(TARGET XPrec::xprec)
    get_target_property(XPREC_INCLUDE_DIRS XPrec::xprec INTERFACE_INCLUDE_DIRECTORIES)
    if(XPREC_INCLUDE_DIRS)
        target_include_directories(libsparseirtests PRIVATE ${XPREC_INCLUDE_DIRS})
    endif()
endif()

add_test(NAME combined_tests COMMAND libsparseirtests -d yes)
