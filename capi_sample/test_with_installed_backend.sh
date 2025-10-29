#!/bin/bash

# Build and run capi_sample against the already installed C++ backend
# This is for CI use where backend is already built and installed

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building and running capi_sample ===${NC}"

# Set SparseIR_DIR to point to the installed backend
export SparseIR_DIR="$(pwd)/../backend/cxx/build/share/cmake/SparseIR"

# Clean build directory
rm -rf ./build
mkdir -p ./build

# Configure
echo -e "${YELLOW}Configuring CMake...${NC}"
cmake -S . -B ./build

# Build
echo -e "${YELLOW}Building sample executables...${NC}"
cmake --build ./build

# Run samples
echo -e "${YELLOW}Running samples...${NC}"
cmake --build ./build --target test

echo -e "${GREEN}=== capi_sample completed successfully ===${NC}"

