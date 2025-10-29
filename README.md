# libsparseir
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SpM-lab/libsparseir)
[![CMake on a single platform](https://github.com/SpM-lab/libsparseir/actions/workflows/CI_cmake.yml/badge.svg)](https://github.com/SpM-lab/libsparseir/actions/workflows/CI_cmake.yml)
[![Create Tag and Release](https://github.com/SpM-lab/libsparseir/actions/workflows/CreateTag.yml/badge.svg)](https://github.com/SpM-lab/libsparseir/actions/workflows/CreateTag.yml)

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

### BLAS Support

BLAS support is **mandatory and always enabled** in this library. BLAS routines are used for performance-critical operations in fitting (`fit_tau`, `fit_matsubara`) and evaluation (`evaluate_tau`, `evaluate_matsubara`).

#### Two Modes for BLAS Provision

The library supports two modes for providing BLAS functions:

1. **Link-time BLAS (default)**: BLAS library is linked at build time
2. **Runtime BLAS registration**: BLAS function pointers are provided at runtime via C-API (used when `SPARSEIR_USE_EXTERN_FBLAS_PTR` is defined)

The choice between these modes is determined at compile time based on the `SPARSEIR_USE_EXTERN_FBLAS_PTR` CMake option.

---

#### Mode 1: Link-time BLAS (Default)

In this mode, a BLAS library (OpenBLAS, Intel MKL, Apple Accelerate, etc.) is linked at build time. The library uses dynamic symbol resolution to automatically detect and use the appropriate BLAS functions at runtime.

**Dynamic Symbol Resolution:**

The library uses `dlopen`/`dlsym` to dynamically resolve Fortran BLAS function symbols at runtime:
- Supports multiple BLAS implementations and symbol naming conventions (`dgemm_`, `dgemm`, `DGEMM_`, `DGEMM`, etc.)
- **ILP64 interfaces are prioritized over LP64 interfaces**:
  1. First attempts to find ILP64 symbols (`dgemm64_`, `dgemm64`, `ZGEMM64_`, etc.)
  2. If ILP64 symbols are found, they are used
  3. Otherwise, falls back to LP64 symbols (`dgemm_`, `dgemm`, `DGEMM_`, etc.)

This automatic detection happens when the library is loaded, without requiring any runtime configuration.

**Building with Link-time BLAS:**

```bash
# Standard build with BLAS auto-detection
mkdir -p build && cd build
cmake ..
cmake --build .

# On Ubuntu with OpenBLAS
sudo apt install libopenblas-dev
cmake ..

# On macOS (uses Accelerate framework automatically)
cmake ..
```

**For ILP64 BLAS:**

If you need ILP64 support for large matrix operations (matrices larger than 2^31 elements), install ILP64-compatible BLAS libraries:

```bash
# Ubuntu with ILP64 OpenBLAS
sudo apt install libopenblas64-0 libopenblas64-dev
cmake .. -DSPARSEIR_USE_BLAS_ILP64=ON
```

**Note**: The `SPARSEIR_USE_BLAS_ILP64` CMake option only affects which BLAS library CMake searches for during configuration (sets `BLA_SIZEOF_INTEGER=8`). At runtime, the library always tries ILP64 symbols first regardless of this option.

**Manual BLAS Library Specification:**

If CMake cannot automatically find the BLAS library, you can specify it manually:

```bash
# For standard LP64 BLAS
cmake .. -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so

# For ILP64 BLAS
cmake .. -DSPARSEIR_USE_BLAS_ILP64=ON \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas64.so.0

# Or use environment variables to help CMake find it
export BLA_VENDOR=OpenBLAS
cmake ..
```

---

#### Mode 2: Runtime BLAS Registration

In this mode (enabled with `-DSPARSEIR_USE_EXTERN_FBLAS_PTR=ON`), the library does not link BLAS at build time. Instead, BLAS function pointers must be registered at runtime before using the library. This mode is primarily used for language bindings (e.g., Python) where BLAS functions are provided by the host environment.

**Building with Runtime BLAS Registration:**

```bash
mkdir -p build && cd build
cmake .. -DSPARSEIR_USE_EXTERN_FBLAS_PTR=ON
cmake --build .
```

**Registering BLAS Functions:**

You must call one of these registration functions before using any BLAS functionality:

**For LP64 interface (32-bit integers):**
```c
void spir_register_dgemm_zgemm_lp64(void* dgemm_fn, void* zgemm_fn);
```

**For ILP64 interface (64-bit integers):**
```c
void spir_register_dgemm_zgemm_ilp64(void* dgemm_fn, void* zgemm_fn);
```

**Important**:
- Only one registration function should be called (either LP64 or ILP64, not both)
- The library will throw a runtime error if BLAS functions are used without prior registration

**Example (Python with ctypes):**

```python
import ctypes
import scipy.linalg.cython_blas as blas

lib = ctypes.CDLL("libsparseir.so")
dgemm_ptr = ctypes.cast(blas.dgemm, ctypes.c_void_p).value
zgemm_ptr = ctypes.cast(blas.zgemm, ctypes.c_void_p).value
lib.spir_register_dgemm_zgemm_lp64(dgemm_ptr, zgemm_ptr)
```

### Debug Logging at runtime

You can also control debug output at runtime using the `SPARSEIR_DEBUG` environment variable:

```bash
export SPARSEIR_DEBUG=1
./your_program
```

## Generating documentation with Doxygen

Install `doxygen` and `graphviz`. Then, run the following command:

```bash
bash generate_docs.sh
```

This will create the `docs/html` directory. Open `docs/html/index.html` with your browser to see it.

## Sample code in C

Please refer [`./sample_c/README.md`](./sample_c/README.md) to learn more.

## Python Bindings

Python bindings are located in the `python/` directory. The bindings use `SPARSEIR_USE_EXTERN_FBLAS_PTR` to register BLAS function pointers from SciPy/Numpy at runtime.

### Testing Python Bindings

To test the Python bindings, use the provided `run_tests.sh` script:

```bash
cd python
./run_tests.sh
```

This script:
1. Cleans up previous build artifacts (copied source files, `.venv`, build cache)
2. Sets up the build environment using `setup_build.py`
3. Installs dependencies and rebuilds the package using `uv sync --refresh`
4. Runs the test suite using `uv run pytest tests/ -v`

The script ensures a clean build environment and automatically handles dependency management with `uv`.

## For developers


### CI/CD

This project uses GitHub Actions for continuous integration and automated releases:

- **CI_cmake.yml**: Runs automated tests on every push and pull request to ensure code quality
- **CreateTag.yml**: Automatically creates tags and releases when version numbers are updated in `include/sparseir/version.h`

### Automated Release Process

The release process is fully automated:

1. Update version numbers using the provided script:
   ```bash
   python update_version.py 0.4.3
   ```
   This automatically updates:
   - `include/sparseir/version.h` (C++ library version)
   - `python/pyproject.toml` (Python package version)

2. Review and commit the changes:
   ```bash
   git diff  # Review changes
   git add -A
   git commit -m "Bump version to 0.4.3"
   ```

3. Push changes to the main branch

4. The GitHub Action will automatically:
   - Extract the version from the header file
   - Check if a tag with that version already exists
   - Create a new tag and release if the version is new
   - Generate release notes automatically
   - Build and publish Python packages to PyPI

#### Version Update Script Usage

The `update_version.py` script provides a convenient way to update versions across all components:

```bash
# Show current versions
python update_version.py

# Update to a new version
python update_version.py 1.0.0

# The script validates version format (x.y.z)
python update_version.py 1.0  # Error: Invalid format
```

**Note**: Always test the build after version updates:
```bash
cd python
pip wheel .  # Test Python package build
```

