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

# libsparseir C-API Documentation

This document describes how to use the C-API of libsparseir. The C-API provides a way to use the sparseir library from C or other languages that can interface with C. All objects are immutable.

## Basic Usage

Please refer [`test/cinterface.cxx`](test/cinterface.cxx) to learn more.


### Basic Working Example
The following example demonstrates how to create a fermionic finite-temperature basis using the logistic kernel,
and perform transformations of a single-variable Green's function between Matsubara frequency and imaginary-time domains.

For fitting, we use the `spir_sampling_fit_XY`, where `X` is the element type of the input data, and `Y` is that of the output data:  `z` corresponds to `c_complex`, `d` to `double`.
The same naming convention is used for evaluation: `spir_sampling_evaluate_XY`.

The logistic kernel is defined as

$
K^\mathrm{L}(\tau, \omega) = \frac{e^{-\tau \omega}}{1 + e^{-\beta\omega}},
$

which corresponds to the normal Fermi-Dirac distribution at temperature $\beta^{-1}$.
For more details, see [SparseIR Tutorial](https://spm-lab.github.io/sparse-ir-tutorial/).

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sparseir/sparseir.h>

// Create a fermionic finite temperature basis
double beta = 10.0;        // Inverse temperature
double omega_max = 10.0;   // Ultraviolet cutoff
double epsilon = 1e-8;     // Accuracy target
spir_fermionic_finite_temp_basis* basis =
    spir_fermionic_finite_temp_basis_new(beta, omega_max, epsilon);

// Create sampling objects for imaginary-time and Matsubara domains
spir_sampling* tau_sampling = spir_fermionic_tau_sampling_new(basis);
spir_sampling* matsubara_sampling = spir_fermionic_matsubara_sampling_new(basis);

// Create Green's function with a pole at 0.5*omega_max
int n_matsubara;
int status = spir_sampling_get_num_points(matsubara_sampling, &n_matsubara);
assert(status == SPIR_COMPUTATION_SUCCESS);

c_complex* g_matsubara = (c_complex*)malloc(n_matsubara * sizeof(c_complex));
int* matsubara_indices = (int*)malloc(n_matsubara * sizeof(int));

// Get Matsubara frequency indices
status = spir_sampling_get_matsubara_points(matsubara_sampling, matsubara_indices);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Set pole position
const double pole_position = 0.0 * omega_max;

// Initialize Green's function in Matsubara frequencies
// G(iω_n) = 1/(iω_n - ε) = -ε/(ω_n^2 + ε^2) - iω_n/(ω_n^2 + ε^2)
for (int i = 0; i < n_matsubara; ++i) {
    assert(abs(matsubara_indices[i]) % 2 == 1); // fermionic Matsubara frequency
    double w_n = matsubara_indices[i] * M_PI / beta;  // Matsubara frequency
    double denominator = w_n * w_n + pole_position * pole_position;
    g_matsubara[i] = -pole_position / denominator - I * w_n / denominator;
}

int target_dim = 0; // target dimension for evaluation and fit

// Matsubara sampling points to basis coefficients
int n_basis;
status = spir_fermionic_finite_temp_basis_get_size(basis, &n_basis);
assert(status == SPIR_COMPUTATION_SUCCESS);
c_complex* g_fit = (c_complex*)malloc(n_basis * sizeof(c_complex));
int dims[1] = {n_matsubara};
status = spir_sampling_fit_zz(matsubara_sampling, SPIR_ORDER_COLUMN_MAJOR,
                             1, dims, target_dim, g_matsubara, g_fit);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Evaluate the basis coefficients at imaginary times
{
    double tau = 0.000001 * beta;
    double expected = -exp(-tau * pole_position) / (1.0 + exp(-beta * pole_position));
    spir_polyvector* u = spir_fermionic_finite_temp_basis_get_u(basis);
    double* uval = (double*)malloc(n_basis * sizeof(double));
    status = spir_evaluate_basis_functions(u, tau, uval);
    assert(status == SPIR_COMPUTATION_SUCCESS);

    double actual = 0.0;
    for (int i = 0; i < n_basis; ++i) {
        actual += creal(g_fit[i]) * uval[i];
    }
    printf("actual: %f, expected: %f\n", actual, expected);
    assert(fabs(actual - expected) < epsilon);

    free(uval);
    spir_destroy_polyvector(u);
}

// Basis coefficients to imaginary-time sampling points
int n_tau;
status = spir_sampling_get_num_points(tau_sampling, &n_tau);
assert(status == SPIR_COMPUTATION_SUCCESS);
c_complex* g_tau = (c_complex*)malloc(n_tau * sizeof(c_complex));
status = spir_sampling_evaluate_zz(tau_sampling, SPIR_ORDER_COLUMN_MAJOR,
                                  1, dims, target_dim, g_fit, g_tau);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Compare with expected result:
//   G(tau) = -exp(-tau * pole_position) / (1 + exp(-beta * pole_position))
double* tau_points = (double*)malloc(n_tau * sizeof(double));
status = spir_sampling_get_tau_points(tau_sampling, tau_points);
assert(status == SPIR_COMPUTATION_SUCCESS);
for (int i = 0; i < n_tau; ++i) {
    double tau = tau_points[i];
    double expected = -exp(-tau * pole_position) / (1.0 + exp(-beta * pole_position));
    assert(fabs(creal(g_tau[i]) - expected) < epsilon);
    assert(fabs(cimag(g_tau[i])) < epsilon);
}

// Imaginary-time sampling points to basis coefficients
c_complex* g_fit2 = (c_complex*)malloc(n_basis * sizeof(c_complex));
status = spir_sampling_fit_zz(tau_sampling, SPIR_ORDER_COLUMN_MAJOR,
                              1, dims, target_dim, g_tau, g_fit2);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Basis coefficients to Matsubara Green's function
c_complex* g_matsubara_reconstructed = (c_complex*)malloc(n_matsubara * sizeof(c_complex));
status = spir_sampling_evaluate_zz(matsubara_sampling, SPIR_ORDER_COLUMN_MAJOR,
                                  1, dims, target_dim, g_fit2, g_matsubara_reconstructed);
assert(status == SPIR_COMPUTATION_SUCCESS);
for (int i = 0; i < n_matsubara; ++i) {
    assert(fabs(creal(g_matsubara_reconstructed[i]) - creal(g_matsubara[i])) < epsilon);
    assert(fabs(cimag(g_matsubara_reconstructed[i]) - cimag(g_matsubara[i])) < epsilon);
}

// Clean up (order is arbitrary)
free(matsubara_indices);
free(g_matsubara);
free(g_fit);
free(g_tau);
spir_destroy_fermionic_finite_temp_basis(basis);
spir_destroy_sampling(tau_sampling);
spir_destroy_sampling(matsubara_sampling);
```


We can create a bosonic basis using the logistic kernel as discussed in the [SparseIR Tutorial](https://spm-lab.github.io/sparse-ir-tutorial/).
This can be achived by replacing `fermionic` by `bosonic` in the above code.