#pragma once

#include <stdbool.h>
#include <complex.h>
#include <stdint.h>

#include "version.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPIR_STATISTICS_FERMIONIC = 1,
    SPIR_STATISTICS_BOSONIC = 0
} spir_statistics_type;


typedef enum {
    SPIR_ORDER_COLUMN_MAJOR = 1,
    SPIR_ORDER_ROW_MAJOR = 0
} spir_order_type;


/* Macro for declaring opaque types and their functions */
#define DECLARE_OPAQUE_TYPE(name)                                              \
    struct _spir_##name;                                                       \
    typedef struct _spir_##name spir_##name;                                  \
                                                                              \
    /* Destroy function */                                                     \
    void spir_destroy_##name(spir_##name *obj);                               \
                                                                              \
    /* Clone function */                                                       \
    spir_##name *spir_clone_##name(const spir_##name *src);                   \
                                                                              \
    /* Check if the shared_ptr is assigned to a valid object */                \
    int spir_is_assigned_##name(const spir_##name *obj);

/* Declare opaque types */
struct _spir_kernel;

DECLARE_OPAQUE_TYPE(kernel);
DECLARE_OPAQUE_TYPE(logistic_kernel);
DECLARE_OPAQUE_TYPE(regularized_bose_kernel);
DECLARE_OPAQUE_TYPE(polyvector);
DECLARE_OPAQUE_TYPE(fermionic_finite_temp_basis);
DECLARE_OPAQUE_TYPE(bosonic_finite_temp_basis);
DECLARE_OPAQUE_TYPE(sampling);
DECLARE_OPAQUE_TYPE(sve_result);
DECLARE_OPAQUE_TYPE(fermionic_dlr);
DECLARE_OPAQUE_TYPE(bosonic_dlr);

/**
 * Kernel
 */
typedef struct _spir_kernel spir_kernel;

/**
 * Function
 */
typedef struct _spir_function spir_function;

/**
 * Basis
 */
//typedef struct _spir_fermionic_finite_temp_basis spir_fermionic_finite_temp_basis;

/**
 * Polynomial vector
 */
//typedef struct _spir_polyvector spir_polyvector;

/**
 * Create new logistic kernel
 */
spir_kernel *spir_logistic_kernel_new(double lambda);

/**
 * Create new regularized bose kernel
 */
spir_kernel *spir_regularized_bose_kernel_new(double lambda);

/**
 * Create new SVE result
 */
spir_sve_result* spir_sve_result_new(const spir_kernel* k, double epsilon);

/**
 * Create new regularized bose kernel
 */
//spir_kernel *spir_regularized_bose_kernel_new(double lambda);

/** Fill [xmin, xmax], [ymin, ymax] with the domain of the kernel. */
int spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                       double *ymin, double *ymax);

/**
 * Takes a kernel `k`, an array `x` of size `nx`, an array `y` of size `ny`
 * and an array `out` of size `nx * ny`. On exit, set
 * `out[ix*ny + iy] = K(x[ix], y[iy])`.
 */
int spir_kernel_matrix(const spir_kernel *k, const double *x, int nx,
                       const double *y, int ny, double *out);

/**
 * Create a new tau sampling object
 */
spir_sampling *spir_fermionic_tau_sampling_new(const spir_fermionic_finite_temp_basis *b);
/**
 * Create a new matsubara sampling object
 */
spir_sampling *spir_fermionic_matsubara_sampling_new(const spir_fermionic_finite_temp_basis *b);

/**
 * Create a new fermionic DLR object
 */
spir_fermionic_dlr *spir_fermionic_dlr_new(const spir_fermionic_finite_temp_basis *b);

/**
 * Create a new bosonic DLR object
 */
spir_bosonic_dlr *spir_bosonic_dlr_new(const spir_bosonic_finite_temp_basis *b);


int spir_sampling_evaluate_dd(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const double *input,          // Input coefficients array
    double *out                    // Output array
    );

int spir_sampling_evaluate_dz(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const double *input,          // Input coefficients array
    std::complex<double> *out                    // Output array
    );

int spir_sampling_evaluate_zz(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const std::complex<double> *input,          // Input coefficients array
    std::complex<double> *out                    // Output array
    );


int spir_sampling_fit_dd(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const double *input,          // Input coefficients array
    double *out                    // Output array
    );

int spir_sampling_fit_zz(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const std::complex<double> *input,          // Input coefficients array
    std::complex<double> *out                    // Output array
    );
/** Destroy instance of kernel */
//void spir_destroy_kernel(spir_kernel *k);

/**
 * Given a kernel k, perform a truncated singular value expansion to precision
 * eps with at most n singular values. `u`, `s`, `v` must be arrays of size
 * at least `n`.
 *
 * On exit, fill `n` with the number of singular values which are significant
 * to a level `eps`, and `u[i]`, `s[i]`, and `v[i]` with the i-th left-singular
 * function, i-th singular value, and i-th right-singular function, respectively.
 */
//int spir_sve(const spir_kernel *k,
             //spir_function *u, double *s, spir_function *v, int *n,
             //double eps);

/** Fill [taumin, taumax] with the domain of the function */
//int spir_func_domain(const spir_function *f, double *taumin, double *taumax);

/**
 * Takes a function `f` and arrays of size `n`: `tau`, `out`.
 * On exit, set `out[n] = f(tau[n])` to double precision.
 */
//int spir_tau_value(const spir_function *f, const double *tau,
                   //double *out, int n);

/**
 * Takes a function `f` and a buffer `f0` of size at least `n`. On exit, set
 * `f0[i]` to the `i`-th root of `f` and set `n` to the number of roots found.
 *
 * Return 0 on success, -1 if more roots were found.
 */
//int spir_tau_roots(const spir_function *f, double *f0, int *n);

/**
 * Takes a function `f` and two arrays of arrays of size `n`: `iw`, `out`.
 * On exit, set `out[n] = FT(f)(iw[n])` to double precision, where `FT` denotes
 * the Fourier transform.
 */
//int spir_iw_value(const spir_function *f, const long *iw,
                  //double _Complex *out, int n);

/**
 * Takes a function `f` and a buffer `f0` of size at least `n`. On exit, set
 * `f0[i]` to the `i`-th sign change of the Fourier transform of `f` and set
 * `n` to the number of roots found.
 *
 * Return 0 on success, -1 if more roots were found.
 */
//int spir_iw_roots(const spir_function *f, long *f0, int n);

// Create new basis
spir_fermionic_finite_temp_basis* spir_fermionic_finite_temp_basis_new(double beta, double omega_max, double epsilon);
spir_bosonic_finite_temp_basis* spir_bosonic_finite_temp_basis_new(double beta, double omega_max, double epsilon);

spir_fermionic_finite_temp_basis *
spir_fermionic_finite_temp_basis_new_with_sve(double beta, double omega_max,
                                             const spir_kernel *k,
                                             const spir_sve_result *sve);

spir_bosonic_finite_temp_basis *
spir_bosonic_finite_temp_basis_new_with_sve(double beta, double omega_max,
                                             const spir_kernel *k,
                                             const spir_sve_result *sve);

// Destroy basis instance
//void spir_destroy_fermionic_basis(spir_fermionic_finite_temp_basis* b);

/**
 * Get basis functions.
 * Returns a polynomial vector that must be freed using spir_destroy_polyvector.
 *
 * @param b The basis
 * @return Polynomial vector, or NULL on error
 */
//spir_polyvector* spir_basis_u(const spir_fermionic_finite_temp_basis* b);

/**
 * Get the size of a polynomial vector.
 *
 * @param v The polynomial vector
 * @return Size of the vector, or -1 on error
 */
//int spir_polyvector_size(const spir_polyvector* v);

/**
 * Destroy a polynomial vector.
 */
//void spir_destroy_polyvector(spir_polyvector* v);

#ifdef __cplusplus
}
#endif
