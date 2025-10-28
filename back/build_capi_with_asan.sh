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
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-w -fsanitize=address -static-libasan" \
  -DCMAKE_CXX_FLAGS="-w -fsanitize=address -static-libasan" \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libasan" \
  -DSPARSEIR_BUILD_FORTRAN=OFF \
  -DSPARSEIR_BUILD_TESTING=OFF \
  -DSPARSEIR_USE_BLAS=ON

# Build (including tests)
cmake --build . --config Release -j 4

# Run tests
ctest --output-on-failure

echo "SparseIR was built with tests successfully."
echo "You can install it using: cd build && cmake --install ." 
