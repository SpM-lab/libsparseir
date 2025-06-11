#!/bin/bash

# Exit on error
set -e

# Create deps directory
mkdir -p deps

# Download and extract Eigen3
if [ ! -d "deps/eigen3" ]; then
    echo "Downloading Eigen3..."
    wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -O eigen3.tar.gz
    tar xzf eigen3.tar.gz -C deps
    mv deps/eigen-3.4.0 deps/eigen3
    rm eigen3.tar.gz
fi

# Download and extract xprec
if [ ! -d "deps/xprec" ]; then
    echo "Downloading xprec..."
    wget -q https://github.com/opencollab/xprec/archive/refs/tags/v1.0.0.tar.gz -O xprec.tar.gz
    tar xzf xprec.tar.gz -C deps
    mv deps/xprec-1.0.0 deps/xprec
    rm xprec.tar.gz
fi

# Build
echo "Building libsparseir..."
make

echo "Done!" 