# Python bindings for libsparseir

This is a low-level binding for the [libsparseir](https://github.com/SpM-lab/libsparseir) library.

## Requirements

- Python >= 3.12
- CMake (for building the C++ library)
- C++11 compatible compiler
- numpy

## Build

### Install Dependencies and Build

```bash
uv sync
```

This will:
- Install Python dependencies (numpy)
- Build the C++ libsparseir library using CMake
- Install the Python package in development mode
