#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Clean build directory to ensure clean state
rm -rf build
mkdir -p build
cd build

# Configure with Fortran bindings
cmake .. \
  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:-$HOME/opt/libsparseir} \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSPARSEIR_BUILD_FORTRAN=ON \
  -DSPARSEIR_BUILD_TESTING=OFF \
  -DBUILD_TESTING=OFF

# Build and install
cmake --build . --config Release
cmake --install .

echo "SparseIR C-API library and Fortran bindings have been built and installed successfully." 
