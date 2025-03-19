#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory
mkdir -p build
cd build

# Configure with tests enabled
cmake .. \
  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:-$HOME/opt/libsparseir} \
  -DSPARSEIR_BUILD_FORTRAN=ON \
  -DSPARSEIR_BUILD_TESTING=ON \
  -DBUILD_TESTING=ON

# Build (including tests)
cmake --build . --config Release

# Run tests
ctest --output-on-failure

echo "SparseIR was built with tests successfully."
echo "You can install it using: cd build && cmake --install ." 