# libsparseir

[![CMake on a single platform](https://github.com/SpM-lab/libsparseir/actions/workflows/CI_cmake.yml/badge.svg)](https://github.com/SpM-lab/libsparseir/actions/workflows/CI_cmake.yml)

> [!WARNING]
> This C++ project is still under construction. Please use other repositories:
> - https://github.com/SpM-lab/sparse-ir
> - https://github.com/SpM-lab/SparseIR.jl
> - https://github.com/SpM-lab/sparse-ir-fortran

## Description

This C++ library provides routines for constructing and working with the intermediate representation of correlation functions. It provides:

- on-the-fly computation of basis functions for arbitrary cutoff Λ
- basis functions and singular values are accurate to full precision
- routines for sparse sampling

We use [tuwien-cms/libxprec](https://github.com/tuwien-cms/libxprec) as a double-double precision arithmetic library.

## Building and Installation

### Dependencies

- **CMake** (>= 3.10)
- **C++ compiler** with C++11 support
- **Fortran compiler** (optional, for Fortran bindings)

All other dependencies (including libxprec) are automatically downloaded and built during the build process using CMake's FetchContent feature. You do not need to install these manually.

### Using Build Scripts

Three build scripts are provided for easy building and installation:

1. **build_capi.sh**: Builds and installs only the C API
   ```bash
   ./build_capi.sh
   ```

2. **build_fortran.sh**: Builds and installs the C API and Fortran bindings
   ```bash
   ./build_fortran.sh
   ```

3. **build_with_tests.sh**: Builds everything including tests
   ```bash
   ./build_with_tests.sh
   # After testing, you can install with:
   cd build && cmake --install .
   ```

By default, all scripts will install to `$HOME/opt/libsparseir`. You can override this by setting the `CMAKE_INSTALL_PREFIX` environment variable:

```bash
CMAKE_INSTALL_PREFIX=/usr/local ./build_capi.sh
```

### Manual Build

If you prefer to build manually, you can use the following commands:

```bash
mkdir -p build
cd build
# For C API only
cmake .. -DSPARSEIR_BUILD_FORTRAN=OFF -DSPARSEIR_BUILD_TESTING=OFF
# For C API and Fortran bindings
cmake .. -DSPARSEIR_BUILD_FORTRAN=ON -DSPARSEIR_BUILD_TESTING=OFF
# For everything including tests
cmake .. -DSPARSEIR_BUILD_FORTRAN=ON -DSPARSEIR_BUILD_TESTING=ON

# Build
cmake --build .

# Install
cmake --install .
```

### Quick Test Build

For a quick test build with all options enabled:

```sh
rm -rf ./build && cmake -S . -B ./build -DSPARSEIR_BUILD_TESTING=ON && cmake --build ./build -j && ./build/test/libsparseirtests
```

### Testing Fortran Bindings

After building with Fortran bindings enabled, you can run the Fortran test:

```bash
cd build
./test_kernel
```

## Generating documentation with Doxygen

Install `doxygen` and `graphviz`. Then, run the following command:

```bash
bash generate_docs.sh
```

This will create the `docs/html` directory. Open `docs/html/index.html` with your browser to see it.

# Sample code in C

Please refer [`sample_c.md`](sample_c.md) to learn more.
