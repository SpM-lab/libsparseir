# libsparseir C++ backend
This is the C++ backend of the libsparseir library. It is implemented in C++11.

## Building and Installation

### Dependencies

- **CMake** (>= 3.10)
- **C++ compiler** with C++11 support
- **BLAS**

All other dependencies (including libxprec) are automatically downloaded and built during the build process using CMake's FetchContent feature. You do not need to install these manually.

### Build & Test

`build_capi_with_tests.sh` builds and tests the C API

### Manual Build

If you prefer to build manually, you can use the following commands:

```bash
mkdir -p build
cd build

# Configure
cmake .. -DSPARSEIR_BUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX="$HOME/opt/libsparseir"

# Build
cmake --build .

# Install
cmake --install .
```


## BLAS Support

BLAS support is **mandatory and always enabled** in this library. BLAS routines are used for performance-critical operations in fitting (`fit_tau`, `fit_matsubara`) and evaluation (`evaluate_tau`, `evaluate_matsubara`).

### Two Modes for BLAS Provision

The library supports two modes for providing BLAS functions:

1. **Link-time BLAS (default)**: BLAS library is linked at build time
2. **Runtime BLAS registration**: BLAS function pointers are provided at runtime via C-API (used when `SPARSEIR_USE_EXTERN_FBLAS_PTR` is defined)

The choice between these modes is determined at compile time based on the `SPARSEIR_USE_EXTERN_FBLAS_PTR` CMake option.

---

### Mode 1: Link-time BLAS (Default)

In this mode, a BLAS library (OpenBLAS, Intel MKL, Apple Accelerate, etc.) is linked at build time. The Fortran BLAS functions (`dgemm_`, `zgemm_`) are called directly.

**ILP64 vs LP64 Interface:**

The library uses compile-time selection to choose between ILP64 (64-bit integers) and LP64 (32-bit integers) BLAS interfaces based on the `SPARSEIR_USE_BLAS_ILP64` CMake option:

- **LP64 (default)**: Uses 32-bit integers (`int`) for matrix dimensions. Suitable for matrices up to 2^31-1 elements.
- **ILP64**: Uses 64-bit integers (`long long`) for matrix dimensions. Required for matrices larger than 2^31-1 elements.

The interface selection is determined at compile time, so you must match the BLAS library interface with the compile-time setting.

**Building with Link-time BLAS:**

```bash
# Standard build with LP64 BLAS (default)
mkdir -p build && cd build
cmake ..
cmake --build .

# On Ubuntu with OpenBLAS (LP64)
sudo apt install libopenblas-dev
cmake ..

# On macOS (uses Accelerate framework automatically, LP64)
cmake ..
```

**For ILP64 BLAS:**

If you need ILP64 support for large matrix operations (matrices larger than 2^31 elements), install ILP64-compatible BLAS libraries and enable the ILP64 option:

```bash
# Ubuntu with ILP64 OpenBLAS
sudo apt install libopenblas64-0 libopenblas64-dev
cmake .. -DSPARSEIR_USE_BLAS_ILP64=ON
```

**Important**: 
- The `SPARSEIR_USE_BLAS_ILP64` CMake option affects both:
  1. Which BLAS library CMake searches for during configuration (sets `BLA_SIZEOF_INTEGER=8` for ILP64 or `BLA_SIZEOF_INTEGER=4` for LP64)
  2. The compiled interface types in the library code (64-bit vs 32-bit integers)
- You must ensure the linked BLAS library matches the compile-time interface selection (ILP64 or LP64)

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

### Mode 2: Runtime BLAS Registration

In this mode (enabled with `-DSPARSEIR_USE_EXTERN_FBLAS_PTR=ON`), the library does not link BLAS at build time. Instead, BLAS function pointers must be registered at runtime before using the library. This mode is primarily used for language bindings (e.g., Python) where BLAS functions are provided by the host environment.

**Building with Runtime BLAS Registration:**

```bash
mkdir -p build && cd build
cmake .. -DSPARSEIR_USE_EXTERN_FBLAS_PTR=ON
cmake --build .
```

**Registering BLAS Functions:**

You must call the appropriate registration function before using any BLAS functionality. The function to call depends on the compile-time setting of `SPARSEIR_USE_BLAS_ILP64`:

- **If built with LP64** (`SPARSEIR_USE_BLAS_ILP64` not set): Call `spir_register_dgemm_zgemm_lp64`
- **If built with ILP64** (`SPARSEIR_USE_BLAS_ILP64=ON`): Call `spir_register_dgemm_zgemm_ilp64`

**C API declarations** (only available when `SPARSEIR_USE_EXTERN_FBLAS_PTR` is defined):

For LP64 interface (32-bit integers):
```c
void spir_register_dgemm_zgemm_lp64(void* dgemm_fn, void* zgemm_fn);
```

For ILP64 interface (64-bit integers):
```c
void spir_register_dgemm_zgemm_ilp64(void* dgemm_fn, void* zgemm_fn);
```

**Important**:
- The registration function must match the compile-time interface selection (ILP64 or LP64)
- Only one registration function should be called (and only the one matching the build configuration)
- The library will throw a runtime error if BLAS functions are used without prior registration
- Function pointers must match the Fortran BLAS signature with the correct integer type (32-bit or 64-bit)

**Example (Python with ctypes for LP64 build):**

```python
import ctypes
import scipy.linalg.cython_blas as blas

lib = ctypes.CDLL("libsparseir.so")
dgemm_ptr = ctypes.cast(blas.dgemm, ctypes.c_void_p).value
zgemm_ptr = ctypes.cast(blas.zgemm, ctypes.c_void_p).value
# For LP64 build:
lib.spir_register_dgemm_zgemm_lp64(dgemm_ptr, zgemm_ptr)
# For ILP64 build, use:
# lib.spir_register_dgemm_zgemm_ilp64(dgemm_ptr, zgemm_ptr)
```

## Debug Logging at runtime

You can also control debug output at runtime using the `SPARSEIR_DEBUG` environment variable:

```bash
export SPARSEIR_DEBUG=1
./your_program
```
