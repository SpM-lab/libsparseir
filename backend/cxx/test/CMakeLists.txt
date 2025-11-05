set(CMAKE_CXX_STANDARD 11)

# Set include paths for tests
set(TEST_INCLUDE_DIRS
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${xprec_SOURCE_DIR}/include
)

# List of all test files
file(GLOB TEST_FILES "*.cxx")

# Create separate test executable for each .cxx file
foreach(TEST_FILE ${TEST_FILES})
    # Get the base name without extension
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    
    # Create executable for this test
    add_executable(${TEST_NAME} ${TEST_FILE})
    
    # Link libraries
    target_link_libraries(${TEST_NAME} PRIVATE Catch2::Catch2WithMain)
    target_link_libraries(${TEST_NAME} PRIVATE SparseIR::sparseir)
    target_link_libraries(${TEST_NAME} PRIVATE Eigen3::Eigen Threads::Threads)
    
    # Add BLAS linking for tests (always enabled)
    if(BLAS_FOUND)
        target_link_libraries(${TEST_NAME} PRIVATE ${BLAS_LIBRARIES})
    endif()
    
    # macOS-specific configuration for tests
    if(APPLE AND ACCELERATE_FRAMEWORK)
        target_link_libraries(${TEST_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
    
    # Add include paths
    if(DEFINED TEST_INCLUDE_DIRS)
        target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDE_DIRS})
    endif()
    
    # Also explicitly include Eigen3 and XPrec include directories
    if(DEFINED EIGEN3_INCLUDE_DIR)
        target_include_directories(${TEST_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
    endif()
    
    if(TARGET XPrec::xprec)
        get_target_property(XPREC_INCLUDE_DIRS XPrec::xprec INTERFACE_INCLUDE_DIRECTORIES)
        if(XPREC_INCLUDE_DIRS)
            target_include_directories(${TEST_NAME} PRIVATE ${XPREC_INCLUDE_DIRS})
        endif()
    endif()
    
    # Add test
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} -d yes)
endforeach() 