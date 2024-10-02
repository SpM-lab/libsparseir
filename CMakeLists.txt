#
# global cmake configuration file for ALPSCore
#

cmake_minimum_required (VERSION 3.1)
project(libsparseir C CXX)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
add_executable(main main.cpp )
target_link_libraries( sample Eigen3::Eigen )
