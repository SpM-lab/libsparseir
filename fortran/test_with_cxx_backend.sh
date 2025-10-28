#!/bin/bash

# Build backend/cxx, install it, then build and run Fortran tests against it
# Directory structure:
#   work_cxx/build_backend  - Build directory for backend/cxx
#   work_cxx/install_backend - Install directory for backend/cxx
#   work_cxx/build_fortran  - Build directory for Fortran bindings
#   work_cxx/build_test     - Build directory for Fortran tests

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

echo -e "${GREEN}=== Testing Fortran with C++ backend ===${NC}"

# Step 1: Build and install backend/cxx (without tests)
echo -e "${YELLOW}Step 1: Building backend/cxx...${NC}"
mkdir -p "$WORK_DIR/build_backend"
cd "$WORK_DIR/build_backend"

cmake "$BACKEND_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DSPARSEIR_BUILD_TESTING=OFF \
    -DSPARSEIR_USE_BLAS=ON

echo -e "${YELLOW}Building backend/cxx...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo -e "${YELLOW}Installing backend/cxx to $INSTALL_DIR...${NC}"
cmake --install .

# Step 2: Build Fortran bindings
echo -e "${YELLOW}Step 2: Building Fortran bindings...${NC}"
cd "$SCRIPT_DIR"
mkdir -p "$WORK_DIR/build_fortran"
cd "$WORK_DIR/build_fortran"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
    -DSPARSEIR_BUILD_TESTING=ON

echo -e "${YELLOW}Building Fortran bindings...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo -e "${YELLOW}Installing Fortran bindings...${NC}"
cmake --install . --prefix "$INSTALL_DIR"

echo -e "${YELLOW}Running Fortran tests...${NC}"
# Set library paths for test execution
export DYLD_LIBRARY_PATH="$INSTALL_DIR/lib:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"

# macOS: Update install_name for Fortran library to find C++ backend
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${YELLOW}Updating install_name for macOS...${NC}"
    # Update install_name for Fortran library to point to installed location
    install_name_tool -id "$INSTALL_DIR/lib/libsparseir_fortran.0.dylib" \
        "$WORK_DIR/build_fortran/libsparseir_fortran.0.dylib" 2>/dev/null || true
    
    # Update rpath for test executables
    for test_bin in "$WORK_DIR/build_fortran/test/"*.exe "$WORK_DIR/build_fortran/test/test_"*; do
        if [ -f "$test_bin" ] && file "$test_bin" | grep -q "Mach-O"; then
            install_name_tool -add_rpath "$INSTALL_DIR/lib" "$test_bin" 2>/dev/null || true
        fi
    done 2>/dev/null || true
fi

ctest --output-on-failure --verbose

echo -e "${GREEN}=== All tests completed successfully ===${NC}"

