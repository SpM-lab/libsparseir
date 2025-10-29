#!/bin/bash

# Build C++ backend with BLAS support and run samples
# Build directory: work_cxx

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building C++ backend with BLAS and running samples ===${NC}"

# Create work_cxx directory structure
BUILD_DIR="work_cxx"
INSTALL_DIR="$BUILD_DIR/install"

mkdir -p "$BUILD_DIR/build_backend"
mkdir -p "$INSTALL_DIR"

# Step 1: Build C++ backend with BLAS
echo -e "${YELLOW}Building C++ backend with BLAS support...${NC}"
cd "$BUILD_DIR/build_backend"

cmake ../../../backend/cxx \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="../../work_cxx/install" \
    -DSPARSEIR_USE_BLAS=ON \
    -DSPARSEIR_BUILD_TESTING=OFF

cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --install .

cd ../..

# Step 2: Set SparseIR_DIR to point to the installed backend
export SparseIR_DIR="$(pwd)/work_cxx/install/share/cmake/SparseIR"

echo -e "${YELLOW}Building samples...${NC}"
cmake -S . -B ./build

echo -e "${YELLOW}Building sample executables...${NC}"
cmake --build build

echo -e "${YELLOW}Running samples...${NC}"
cmake --build build --target test

echo -e "${GREEN}=== All samples completed successfully ===${NC}"

