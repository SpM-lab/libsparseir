include(FetchContent)

# Require Catch2 for tests
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
    FIND_PACKAGE_ARGS 3
    )

# Goes through find_package first
FetchContent_MakeAvailable(Catch2)

add_executable(libsparseirtests
    example.cxx
)

target_link_libraries(libsparseirtests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(libsparseirtests PRIVATE SparseIR::sparseir)

#if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.12")
    #set_property(TARGET tests PROPERTY CXX_STANDARD 20)
#else()
    #set_property(TARGET tests PROPERTY CXX_STANDARD 17)
#endif()
add_test(tests libsparseirtests)