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
- **Eigen3** (>= 3.4)
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

# libsparseir C-API Documentation

This document describes how to use the C-API of libsparseir. The C-API provides a way to use the sparseir library from C or other languages that can interface with C. All objects are immutable.

## Basic Usage

### Kernel Creation and Domain

```c
#include <sparseir/sparseir.h>

// Create kernels for different statistics
spir_kernel* fermionic_kernel = spir_logistic_kernel_new(9.0);
spir_kernel* bosonic_kernel = spir_regularized_bose_kernel_new(9.0);

// Get kernel domain
double xmin, xmax, ymin, ymax;
spir_kernel_domain(fermionic_kernel, &xmin, &xmax, &ymin, &ymax);

// Clean up
spir_destroy_kernel(fermionic_kernel);
spir_destroy_kernel(bosonic_kernel);
```

### Basis Construction and Sampling

```c
#include <sparseir/sparseir.h>

// Create a fermionic finite temperature basis
double beta = 10.0;        // Inverse temperature
double omega_max = 10.0;   // Frequency cutoff
double epsilon = 1e-8;     // Accuracy target
spir_fermionic_finite_temp_basis* basis =
    spir_fermionic_finite_temp_basis_new(beta, omega_max, epsilon);

// Create sampling objects for different domains
spir_sampling* tau_sampling = spir_fermionic_tau_sampling_new(basis);
spir_sampling* matsubara_sampling = spir_fermionic_matsubara_sampling_new(basis);

// Clean up
spir_destroy_fermionic_finite_temp_basis(basis);
spir_destroy_sampling(tau_sampling);
spir_destroy_sampling(matsubara_sampling);
```

### Sampling Operations

```c
#include <sparseir/sparseir.h>

// Create basis and sampling objects
double beta = 10.0;
double omega_max = 10.0;
double epsilon = 1e-8;
spir_fermionic_finite_temp_basis* basis =
    spir_fermionic_finite_temp_basis_new(beta, omega_max, epsilon);
spir_sampling* tau_sampling = spir_fermionic_tau_sampling_new(basis);

// Example of evaluating basis coefficients at sampling points
int32_t ndim = 2;
int32_t input_dims[] = {3, 4};  // Example dimensions
int32_t target_dim = 0;
double input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
double output[12];

// Real to real transformation
spir_sampling_evaluate_dd(
    tau_sampling,
    SPIR_ORDER_COLUMN_MAJOR,
    ndim,
    input_dims,
    target_dim,
    input,
    output
);

// Fit values back to basis coefficients
spir_sampling_fit_dd(
    tau_sampling,
    SPIR_ORDER_COLUMN_MAJOR,
    ndim,
    input_dims,
    target_dim,
    output,
    input  // Original coefficients should be recovered
);

// Clean up
spir_destroy_fermionic_finite_temp_basis(basis);
spir_destroy_sampling(tau_sampling);
```

### Discrete Lehmann Representation (DLR)

```c
#include <sparseir/sparseir.h>

// First, create a kernel and compute its SVE
double lambda = 10.0;
spir_kernel* kernel = spir_logistic_kernel_new(lambda);

double epsilon = 1e-8;
spir_sve_result* sve = spir_sve_result_new(kernel, epsilon);

// Create basis with pre-computed SVE
double beta = 10.0;
double omega_max = 10.0;
spir_fermionic_finite_temp_basis* basis =
    spir_fermionic_finite_temp_basis_new_with_sve(beta, omega_max, kernel, sve);

// Create a DLR object
spir_fermionic_dlr* dlr = spir_fermionic_dlr_new(basis);

// Get fitting matrix dimensions
size_t rows = spir_fermionic_dlr_fitmat_rows(dlr);
size_t cols = spir_fermionic_dlr_fitmat_cols(dlr);
// Prepare data for transformation
int32_t ndim = 1;
int32_t input_dims[] = {(int32_t)cols};  // Size must match the number of IR basis functions

// Allocate arrays with proper sizes
double* ir_coeffs = (double*)malloc(cols * sizeof(double));
double* dlr_coeffs = (double*)malloc(rows * sizeof(double));
double* recovered_ir = (double*)malloc(cols * sizeof(double));

// Initialize IR coefficients with some values
for (size_t i = 0; i < cols; i++) {
    ir_coeffs[i] = 1.0 / (1.0 + i);
}

// Transform from IR to DLR representation
// This line causes an error
int status = spir_fermionic_dlr_from_IR(
    dlr,
    SPIR_ORDER_COLUMN_MAJOR,
    ndim,
    input_dims,
    ir_coeffs,
    dlr_coeffs
);
/*

// Transform back to IR representation
status = spir_fermionic_dlr_to_IR(
    dlr,
    SPIR_ORDER_COLUMN_MAJOR,
    ndim,
    input_dims,
    dlr_coeffs,
    recovered_ir
);

// Clean up
delete[] ir_coeffs;
delete[] dlr_coeffs;
delete[] recovered_ir;
spir_destroy_fermionic_dlr(dlr);
spir_destroy_fermionic_finite_temp_basis(basis);
spir_destroy_sve_result(sve);
spir_destroy_kernel(kernel);
*/
```

### Complex Number Operations

```c
#include <sparseir/sparseir.h>
#include <complex.h>

// Create basis and sampling objects
double beta = 10.0;
double omega_max = 10.0;
double epsilon = 1e-8;
spir_fermionic_finite_temp_basis* basis =
    spir_fermionic_finite_temp_basis_new(beta, omega_max, epsilon);
spir_sampling* matsubara_sampling = spir_fermionic_matsubara_sampling_new(basis);

// Setup dimensions and data
int32_t ndim = 2;
int32_t input_dims[] = {3, 4};
int32_t target_dim = 0;

// Initialize complex input data
c_complex complex_input[12];
c_complex complex_output[12];
for (int i = 0; i < 12; i++) {
    // Complex numbers with real and imaginary parts
    complex_input[i] = c_complex{(double)i, (double)i};
}

// Complex to complex transformation
spir_sampling_evaluate_zz(
    matsubara_sampling,
    SPIR_ORDER_COLUMN_MAJOR,
    ndim,
    input_dims,
    target_dim,
    complex_input,
    complex_output
);

// Fit complex values back to basis coefficients
c_complex recovered_input[12];
spir_sampling_fit_zz(
    matsubara_sampling,
    SPIR_ORDER_COLUMN_MAJOR,
    ndim,
    input_dims,
    target_dim,
    complex_output,
    recovered_input
);

// Clean up
spir_destroy_fermionic_finite_temp_basis(basis);
spir_destroy_sampling(matsubara_sampling);
```

# libsparseir Fortran Bindings Documentation

This document describes how to use the Fortran bindings of libsparseir. The Fortran bindings provide a way to use the sparseir library from Fortran. All objects are immutable.

## Basic Usage

```fortran
use sparseir
implicit none

type(spir_kernel) :: kernel
real(8) :: lambda, xmin, xmax, ymin, ymax
integer :: stat

! Create a logistic kernel
lambda = 9.0d0
kernel = spir_logistic_kernel_new(lambda)

! Get kernel domain
stat = spir_kernel_domain(kernel, xmin, xmax, ymin, ymax)
```