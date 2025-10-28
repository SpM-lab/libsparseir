set(CMAKE_CXX_STANDARD 11)

# C-API integration test
add_executable(cinterface_integration cinterface_integration.cxx)

target_include_directories(cinterface_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../cxx/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../cxx/test
    ${EIGEN3_INCLUDE_DIR}
    ${xprec_SOURCE_DIR}/include
)

target_link_libraries(cinterface_integration PRIVATE 
    Catch2::Catch2WithMain
    SparseIR::sparseir
    Eigen3::Eigen 
    Threads::Threads
)

if(SPARSEIR_USE_BLAS AND BLAS_FOUND)
    target_link_libraries(cinterface_integration PRIVATE ${BLAS_LIBRARIES})
endif()

if(APPLE AND ACCELERATE_FRAMEWORK)
    target_link_libraries(cinterface_integration PRIVATE ${ACCELERATE_FRAMEWORK})
endif()

add_test(NAME cinterface_integration COMMAND cinterface_integration -d yes)

