#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Clean build directory to ensure clean state
rm -rf build
mkdir -p build
cd build

# Configure with maximum optimization for benchmarking
cmake .. \
  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:-$HOME/opt/libsparseir} \
  -DSPARSEIR_BUILD_FORTRAN=ON \
  -DSPARSEIR_BUILD_TESTING=OFF \
  -DSPARSEIR_USE_BLAS=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -g0 -march=native" \
  -DCMAKE_EXE_LINKER_FLAGS="-flto" \
  -DCMAKE_VERBOSE_MAKEFILE=ON

# Build with maximum parallelization
cmake --build . --config Release -- -j $(nproc)

# Install
cmake --install .

echo "Optimized SparseIR library has been built and installed successfully."
echo "Debug symbols removed, maximum optimization enabled."
