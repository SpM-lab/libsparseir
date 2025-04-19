include(FetchContent)

set(CMAKE_CXX_STANDARD 11)

# Set include paths for tests
set(TEST_INCLUDE_DIRS
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${xprec_SOURCE_DIR}/include
)

# Also make sure XPrec is available to tests
if(TARGET XPrec::xprec)
    get_target_property(XPREC_INCLUDE_DIRS XPrec::xprec INTERFACE_INCLUDE_DIRECTORIES)
    if(XPREC_INCLUDE_DIRS)
        list(APPEND TEST_INCLUDE_DIRS ${XPREC_INCLUDE_DIRS})
    endif()
endif()

# Find Python
find_package(Python3 REQUIRED)

# Generate test file from README
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/readme_samples_test.cpp
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_test.py
            ${CMAKE_SOURCE_DIR}/README.md
            ${CMAKE_CURRENT_BINARY_DIR}/readme_samples_test.cpp
    DEPENDS ${CMAKE_SOURCE_DIR}/README.md
            ${CMAKE_CURRENT_SOURCE_DIR}/generate_test.py
    COMMENT "Generating test file from README.md"
)

# Add test executable
add_executable(readme_samples_test
    ${CMAKE_CURRENT_BINARY_DIR}/readme_samples_test.cpp
)

target_link_libraries(readme_samples_test
    PRIVATE sparseir
    PRIVATE Catch2::Catch2WithMain
)

# Add test
add_test(NAME readme_samples_test
    COMMAND readme_samples_test
)

# Add include paths from parent scope if available
if(DEFINED TEST_INCLUDE_DIRS)
    target_include_directories(readme_samples_test PRIVATE ${TEST_INCLUDE_DIRS})
endif()

# Also explicitly include Eigen3 and XPrec include directories
if(DEFINED EIGEN3_INCLUDE_DIR)
    target_include_directories(readme_samples_test PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

if(TARGET XPrec::xprec)
    get_target_property(XPREC_INCLUDE_DIRS XPrec::xprec INTERFACE_INCLUDE_DIRECTORIES)
    if(XPREC_INCLUDE_DIRS)
        target_include_directories(readme_samples_test PRIVATE ${XPREC_INCLUDE_DIRS})
    endif()
endif() 
