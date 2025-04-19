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

Please refer [`test/cinterface.cxx`](test/cinterface.cxx) to learn more.

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
    input
);

// Clean up
spir_destroy_fermionic_finite_temp_basis(basis);
spir_destroy_sampling(tau_sampling);
```

### Discrete Lehmann Representation (DLR)

```c
#include <sparseir/sparseir.h>
#include <stdlib.h> // rand

double beta = 100.0;
double wmax = 1.0;
double epsilon = 1e-12;

spir_fermionic_finite_temp_basis *basis =
    spir_fermionic_finite_temp_basis_new(beta, wmax, epsilon);

spir_fermionic_dlr *dlr = spir_fermionic_dlr_new(basis);

int npoles = 10;
double poles[npoles];
double coeffs[npoles];
for (int i = 0; i < npoles; i++) {
    double r = (double)rand() / (double)RAND_MAX;
    poles[i] = wmax * (2.0 * r - 1.0);
    r = (double)rand() / (double)RAND_MAX;
    coeffs[i] = 2.0 * r - 1.0;
}

spir_fermionic_dlr *dlr_with_poles =
    spir_fermionic_dlr_new_with_poles(basis, npoles, poles);
size_t fitmat_rows = spir_fermionic_dlr_fitmat_rows(dlr_with_poles);
size_t fitmat_cols = spir_fermionic_dlr_fitmat_cols(dlr_with_poles);

double *Gl = (double *)malloc(fitmat_rows * sizeof(double));
int32_t to_ir_input_dims[1] = {npoles};
int status_to_IR =
    spir_fermionic_dlr_to_IR(dlr_with_poles, SPIR_ORDER_COLUMN_MAJOR, 1,
                                to_ir_input_dims, coeffs, Gl);

double *g_dlr = (double *)malloc(fitmat_rows * sizeof(double));
int32_t from_ir_input_dims[1] = {static_cast<int32_t>(fitmat_rows)};
int status_from_IR = spir_fermionic_dlr_from_IR(
    dlr, SPIR_ORDER_COLUMN_MAJOR, 1, from_ir_input_dims, Gl, g_dlr);

// Clean up
free(Gl);
free(g_dlr);
spir_destroy_fermionic_finite_temp_basis(basis);
spir_destroy_fermionic_dlr(dlr);
spir_destroy_fermionic_dlr(dlr_with_poles);
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