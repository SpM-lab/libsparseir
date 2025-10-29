# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-src")
  file(MAKE_DIRECTORY "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-src")
endif()
file(MAKE_DIRECTORY
  "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-build"
  "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-subbuild/xprec-populate-prefix"
  "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-subbuild/xprec-populate-prefix/tmp"
  "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-subbuild/xprec-populate-prefix/src/xprec-populate-stamp"
  "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-subbuild/xprec-populate-prefix/src"
  "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-subbuild/xprec-populate-prefix/src/xprec-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-subbuild/xprec-populate-prefix/src/xprec-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/hiroshi/projects/sparse-ir/libsparseir/capi_benchmark/work_cxx_blas/build_backend/_deps/xprec-subbuild/xprec-populate-prefix/src/xprec-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
