#!/bin/bash

# Build backend/cxx, install it, then build and run capi_test against it
# Directory structure:
#   work_cxx/build_backend  - Build directory for backend/cxx
#   work_cxx/install_backend - Install directory for backend/cxx
#   work_cxx/build_test     - Build directory for capi_test

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$SCRIPT_DIR/work_cxx"
BACKEND_DIR="$SCRIPT_DIR/../backend/cxx"
INSTALL_DIR="$WORK_DIR/install_backend"

echo -e "${GREEN}=== Testing capi_test with C++ backend ===${NC}"

# Step 1: Build and install backend/cxx (without tests)
echo -e "${YELLOW}Step 1: Building backend/cxx...${NC}"
rm -rf "$WORK_DIR/build_backend"
mkdir -p "$WORK_DIR/build_backend"
cd "$WORK_DIR/build_backend"

cmake "$BACKEND_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DSPARSEIR_BUILD_TESTING=OFF \
    ${SPARSEIR_USE_BLAS_ILP64:+-DSPARSEIR_USE_BLAS_ILP64=ON}

echo -e "${YELLOW}Building backend/cxx...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo -e "${YELLOW}Installing backend/cxx to $INSTALL_DIR...${NC}"
cmake --install .

# Step 2: Build and test capi_test
echo -e "${YELLOW}Step 2: Building capi_test...${NC}"
cd "$SCRIPT_DIR"
mkdir -p "$WORK_DIR/build_test"
cd "$WORK_DIR/build_test"

# Set SDK path explicitly to avoid CMake auto-detection issues
#export SDKROOT=$(xcrun --show-sdk-path)

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR"

echo -e "${YELLOW}Building capi_test...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo -e "${YELLOW}Running capi_test...${NC}"
ctest --output-on-failure --verbose

echo -e "${GREEN}=== All tests completed successfully ===${NC}"

