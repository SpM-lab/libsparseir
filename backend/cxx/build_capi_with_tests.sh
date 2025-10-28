#!/bin/bash

# Build backend/cxx and run C++ tests
# Build directory: build

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building backend/cxx with tests ===${NC}"

# Create build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

# Configure CMake
echo -e "${YELLOW}Configuring CMake...${NC}"
cd "$BUILD_DIR"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPARSEIR_BUILD_TESTING=ON \
    -DSPARSEIR_USE_BLAS=ON

# Build
echo -e "${YELLOW}Building...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run tests
echo -e "${YELLOW}Running C++ tests...${NC}"
ctest --output-on-failure --verbose

echo -e "${GREEN}=== Build and test completed successfully ===${NC}"

