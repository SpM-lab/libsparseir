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


### Fermionic basis with logistic kernel

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sparseir/sparseir.h>

typedef double _Complex c_complex;

int32_t status;

// Create a fermionic finite temperature basis
double beta = 10.0;        // Inverse temperature
double omega_max = 10.0;   // Ultraviolet cutoff
double epsilon = 1e-8;     // Accuracy target

// Create a logistic kernel
spir_kernel* kernel = spir_logistic_kernel_new(beta * omega_max);

// Create a pre-computed SVE result
spir_sve_result* sve = spir_sve_result_new(kernel, epsilon);

// Create a fermionic finite temperature basis with pre-computed SVE result
// Use SPIR_STATISTICS_BOSONIC for bosonic basis
spir_finite_temp_basis* basis =
    spir_finite_temp_basis_new_with_sve(SPIR_STATISTICS_FERMIONIC, beta, omega_max, kernel, sve);

int n_basis;
status = spir_finite_temp_basis_get_size(basis, &n_basis);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Evaluate the basis functions at a given tau point
spir_funcs* u;
status = spir_finite_temp_basis_get_u(basis, &u);
assert(status == SPIR_COMPUTATION_SUCCESS);
double tau = 0.5 * beta;
double* uval = (double*)malloc(n_basis * sizeof(double));
status = spir_evaluate_funcs(u, tau, uval);
assert(status == SPIR_COMPUTATION_SUCCESS);
for (int i = 0; i < n_basis; ++i) {
    printf("u[%d] = %f\n", i, uval[i]);
}

// Evaluate the basis functions at a given omega point
spir_funcs* v;
status = spir_finite_temp_basis_get_v(basis, &v);
assert(status == SPIR_COMPUTATION_SUCCESS);
double omega = 0.5 * omega_max;
double* vval = (double*)malloc(n_basis * sizeof(double));
status = spir_evaluate_funcs(v, omega, vval);
assert(status == SPIR_COMPUTATION_SUCCESS);
for (int i = 0; i < n_basis; ++i) {
    printf("v[%d] = %f\n", i, vval[i]);
}

// Evaluate the basis functions at given Matsubara frequencies
int n_freqs = 10;
int* matsubara_freq_indices = (int*)malloc(n_freqs * sizeof(int));
for (int i = 0; i < n_freqs; ++i) {
    matsubara_freq_indices[i] = 2 * i + 1; // fermionic Matsubara frequency
}

c_complex* uhat_val = (c_complex*)malloc(n_basis * n_freqs * sizeof(c_complex));
spir_matsubara_funcs* uhat;
status = spir_finite_temp_basis_get_uhat(basis, &uhat);
assert(status == SPIR_COMPUTATION_SUCCESS);
status = spir_evaluate_matsubara_funcs(uhat, SPIR_ORDER_COLUMN_MAJOR, n_freqs, matsubara_freq_indices, uhat_val);
assert(status == SPIR_COMPUTATION_SUCCESS);

// Clean up (in arbitrary order)
free(uval);
free(vval);
spir_destroy_funcs(u);
spir_destroy_funcs(v);
spir_destroy_matsubara_funcs(uhat);
spir_destroy_finite_temp_basis(basis);
```

### Bosonic basis

A bosonic basis can be created by replacing `SPIR_STATISTICS_FERMIONIC` with `SPIR_STATISTICS_BOSONIC` in the fermionic basis construction code.
We can use either `spir_logistic_kernel_new` or `spir_regularized_bose_kernel_new` for a bosonic basis.
For the definitions, we refer to the [SparseIR Tutorial](https://spm-lab.github.io/sparse-ir-tutorial/).

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sparseir/sparseir.h>

typedef double _Complex c_complex;

int32_t status;

// Create a bosonic finite temperature basis
double beta = 10.0;        // Inverse temperature
double omega_max = 10.0;   // Ultraviolet cutoff
double epsilon = 1e-8;     // Accuracy target

// Create a logistic kernel
spir_kernel* logistic_kernel = spir_logistic_kernel_new(beta * omega_max);

// Create a pre-computed SVE result
spir_sve_result* sve_logistic = spir_sve_result_new(logistic_kernel, epsilon);

// Create fermionic and bosonic finite temperature bases with pre-computed SVE result
spir_finite_temp_basis* basis_fermionic =
    spir_finite_temp_basis_new_with_sve(SPIR_STATISTICS_FERMIONIC, beta, omega_max, logistic_kernel, sve_logistic);

spir_finite_temp_basis* basis_bosonic =
    spir_finite_temp_basis_new_with_sve(SPIR_STATISTICS_BOSONIC, beta, omega_max, logistic_kernel, sve_logistic);

// Create a regularized bosonic basis
spir_kernel* regularized_kernel = spir_regularized_bose_kernel_new(beta * omega_max);
spir_sve_result* sve_regularized = spir_sve_result_new(regularized_kernel, epsilon);
spir_finite_temp_basis* basis_regularized =
    spir_finite_temp_basis_new_with_sve(SPIR_STATISTICS_BOSONIC, beta, omega_max, regularized_kernel, sve_regularized);

// Clean up (in arbitrary order)
spir_destroy_kernel(logistic_kernel);
spir_destroy_sve_result(sve_logistic);
spir_destroy_kernel(regularized_kernel);
spir_destroy_sve_result(sve_regularized);
spir_destroy_finite_temp_basis(basis_fermionic);
spir_destroy_finite_temp_basis(basis_bosonic);
spir_destroy_finite_temp_basis(basis_regularized);
```

### Sparse Sampling
The following example demonstrates how to transform a single-variable Green's function between Matsubara frequency and imaginary-time domains.

For fitting, we use `spir_sampling_fit_XY`, where `X` and `Y` specify the data types of the input and output data respectively: `z` represents `double _Complex` and `d` represents `double`.
The same naming convention applies to evaluation functions: `spir_sampling_evaluate_XY`.

The fitting and evaluation functions support multi-dimensional input data, where one dimension represents the number of sampling points or basis functions, and additional dimensions can represent quantities like momentum, spin, or orbital indices.
The dimension to transform is specified by the `target_dim` argument.
The transformed dimension maintains its position in the output array.
The memory layout of the input and output arrays is specified by the `order` argument, which can be either `SPIR_ORDER_COLUMN_MAJOR` or `SPIR_ORDER_ROW_MAJOR`.

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <sparseir/sparseir.h>

typedef double _Complex c_complex;

// Create a fermionic finite temperature basis
double beta = 10.0;        // Inverse temperature
double omega_max = 10.0;   // Ultraviolet cutoff
double epsilon = 1e-8;     // Accuracy target
spir_finite_temp_basis* basis =
    spir_finite_temp_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon); // default choice of kernel is logistic kernel

// Create sampling objects for imaginary-time and Matsubara domains
spir_sampling* tau_sampling = spir_tau_sampling_new(basis);
spir_sampling* matsubara_sampling = spir_matsubara_sampling_new(basis);

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
// G(iω_n) = 1/(iω_n - ε)
for (int i = 0; i < n_matsubara; ++i) {
    assert(abs(matsubara_indices[i]) % 2 == 1); // fermionic Matsubara frequency
    g_matsubara[i] = 1.0 / (I * matsubara_indices[i] * M_PI / beta - pole_position);
}

int target_dim = 0; // target dimension for evaluation and fit

// Matsubara sampling points to basis coefficients
int n_basis;
status = spir_finite_temp_basis_get_size(basis, &n_basis);
assert(status == SPIR_COMPUTATION_SUCCESS);
c_complex* g_fit = (c_complex*)malloc(n_basis * sizeof(c_complex));
int dims[1] = {n_matsubara};
status = spir_sampling_fit_zz(matsubara_sampling, SPIR_ORDER_COLUMN_MAJOR,
                             1, dims, target_dim, g_matsubara, g_fit);
assert(status == SPIR_COMPUTATION_SUCCESS);

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
free(g_fit2);
free(g_tau);
spir_destroy_finite_temp_basis(basis);
spir_destroy_sampling(tau_sampling);
spir_destroy_sampling(matsubara_sampling);
```

## Calling from C++
We have to be careful about the interoperability between `double _Complex` and `std::complex<double>`.
This library uses the C99 `_Complex` type. Although its memory layout is often similar to that of `std::complex<double>`,
this compatibility is not guaranteed by the C++ standard.

Therefore, when calling C functions from C++ or exchanging data between these types,
it is _highly_ recommended to explicitly convert values using safe accessors rather than relying on type punning or `memcpy`.

The following code demonstrates how to convert between `double _Complex` and `std::complex<double>` safely.

```cpp
#include <complex>
#include <complex.h>
#include <vector>

// Convert from C99 double _Complex to std::complex<double>
std::vector<std::complex<double>> convert_to_cpp(const std::vector<double _Complex>& c_array) {
    std::vector<std::complex<double>> cpp_array(c_array.size());
    for (std::size_t i = 0; i < c_array.size(); ++i) {
        cpp_array[i] = std::complex<double>(creal(c_array[i]), cimag(c_array[i]));
    }
    return cpp_array;
}

// Convert from std::complex<double> to C99 double _Complex
std::vector<double _Complex> convert_to_c(const std::vector<std::complex<double>>& cpp_array) {
    std::vector<double _Complex> c_array(cpp_array.size());
    for (std::size_t i = 0; i < cpp_array.size(); ++i) {
        c_array[i] = cpp_array[i].real() + cpp_array[i].imag() * I;
    }
    return c_array;
}
```

To pass a double _Complex array to a C function, use `c_array.data()` to obtain a pointer to the first element.