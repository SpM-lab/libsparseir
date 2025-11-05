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
BACKEND_BUILD_DIR="$BACKEND_DIR/build"

echo -e "${GREEN}=== Testing capi_test with C++ backend ===${NC}"

# Step 1: Build and install backend/cxx (without tests)
# Use existing build if available, otherwise build fresh
echo -e "${YELLOW}Step 1: Building backend/cxx...${NC}"

if [ -d "$BACKEND_BUILD_DIR" ] && ([ -f "$BACKEND_BUILD_DIR/libsparseir.dylib" ] || [ -f "$BACKEND_BUILD_DIR/libsparseir.so" ]); then
    echo -e "${YELLOW}Using existing backend build at $BACKEND_BUILD_DIR...${NC}"
    mkdir -p "$INSTALL_DIR/lib"
    mkdir -p "$INSTALL_DIR/include/sparseir"
    
    # Copy library files
    cp "$BACKEND_BUILD_DIR"/libsparseir*.dylib "$INSTALL_DIR/lib/" 2>/dev/null || \
    cp "$BACKEND_BUILD_DIR"/libsparseir*.so "$INSTALL_DIR/lib/" 2>/dev/null || \
    cp "$BACKEND_BUILD_DIR"/libsparseir*.a "$INSTALL_DIR/lib/" 2>/dev/null || true
    
    # Copy header files
    cp "$BACKEND_DIR/include/sparseir"/*.h "$INSTALL_DIR/include/sparseir/" 2>/dev/null || true
    
    # Create cmake config if needed
    mkdir -p "$INSTALL_DIR/share/cmake/SparseIR"
    if [ ! -f "$INSTALL_DIR/share/cmake/SparseIR/SparseIRConfig.cmake" ]; then
        cat > "$INSTALL_DIR/share/cmake/SparseIR/SparseIRConfig.cmake" << 'EOF'
set(SparseIR_FOUND TRUE)
find_library(SPARSEIR_LIBRARY sparseir PATHS "${CMAKE_CURRENT_LIST_DIR}/../../../lib" NO_DEFAULT_PATH)
if(SPARSEIR_LIBRARY)
    add_library(SparseIR::sparseir SHARED IMPORTED)
    set_target_properties(SparseIR::sparseir PROPERTIES
        IMPORTED_LOCATION "${SPARSEIR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../../include"
    )
endif()
EOF
    fi
else
    echo -e "${YELLOW}Building backend/cxx from scratch...${NC}"
    mkdir -p "$WORK_DIR/build_backend"
    cd "$WORK_DIR/build_backend"

    cmake "$BACKEND_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DSPARSEIR_BUILD_TESTING=OFF \
        ${SPARSEIR_USE_BLAS_ILP64:+-DSPARSEIR_USE_BLAS_ILP64=ON} || {
        echo -e "${RED}Failed to configure backend. Trying to use existing build...${NC}"
        if [ -d "$BACKEND_BUILD_DIR" ]; then
            mkdir -p "$INSTALL_DIR/lib"
            mkdir -p "$INSTALL_DIR/include/sparseir"
            cp "$BACKEND_BUILD_DIR"/libsparseir*.dylib "$INSTALL_DIR/lib/" 2>/dev/null || \
            cp "$BACKEND_BUILD_DIR"/libsparseir*.so "$INSTALL_DIR/lib/" 2>/dev/null || true
            cp "$BACKEND_DIR/include/sparseir"/*.h "$INSTALL_DIR/include/sparseir/" 2>/dev/null || true
        else
            exit 1
        fi
    }

    echo -e "${YELLOW}Building backend/cxx...${NC}"
    cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) || {
        echo -e "${RED}Failed to build backend. Trying to use existing build...${NC}"
        if [ -d "$BACKEND_BUILD_DIR" ]; then
            mkdir -p "$INSTALL_DIR/lib"
            mkdir -p "$INSTALL_DIR/include/sparseir"
            cp "$BACKEND_BUILD_DIR"/libsparseir*.dylib "$INSTALL_DIR/lib/" 2>/dev/null || \
            cp "$BACKEND_BUILD_DIR"/libsparseir*.so "$INSTALL_DIR/lib/" 2>/dev/null || true
            cp "$BACKEND_DIR/include/sparseir"/*.h "$INSTALL_DIR/include/sparseir/" 2>/dev/null || true
        else
            exit 1
        fi
    }

    echo -e "${YELLOW}Installing backend/cxx to $INSTALL_DIR...${NC}"
    cmake --install . || {
        echo -e "${YELLOW}Install failed, but continuing with build directory...${NC}"
    }
fi

# Step 2: Build and test capi_test
echo -e "${YELLOW}Step 2: Building capi_test...${NC}"
cd "$SCRIPT_DIR"
mkdir -p "$WORK_DIR/build_test"
cd "$WORK_DIR/build_test"

SDK_PATH=$(xcrun --show-sdk-path 2>/dev/null || echo "")
if [ -n "$SDK_PATH" ]; then
    CMAKE_OSX_SYSROOT_ARG="-DCMAKE_OSX_SYSROOT=$SDK_PATH"
else
    CMAKE_OSX_SYSROOT_ARG="-DCMAKE_OSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion | cut -d. -f1-2)"
fi

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$INSTALL_DIR" \
    $CMAKE_OSX_SYSROOT_ARG

echo -e "${YELLOW}Building capi_test...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo -e "${YELLOW}Running capi_test...${NC}"
export DYLD_LIBRARY_PATH="$INSTALL_DIR/lib:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"
ctest --output-on-failure --verbose

echo -e "${GREEN}=== All tests completed successfully ===${NC}"

