#pragma once

#include <stdbool.h>
#include <complex.h>

#include "version.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types
struct _spir_kernel;
struct _spir_sve;

/**
 * Kernel
 */
typedef struct _spir_kernel spir_kernel;

/**
 * Function
 */
typedef struct _spir_function spir_function;


/** Make new logistic kernel for given UV cutoff lambda */
spir_kernel *spir_logistic_kernel(double lambda);

/** Make new regularized Bose kernel for given UV cutoff lambda. */
spir_kernel *spir_regularized_bose_kernel(double lambda);

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

/** Destroy instance of kernel */
void spir_destroy_kernel(spir_kernel *k);

/**
 * Given a kernel k, perform a truncated singular value expansion to precision
 * eps with at most n singular values. `u`, `s`, `v` must be arrays of size
 * at least `n`.
 *
 * On exit, fill `n` with the number of singular values which are significant
 * to a level `eps`, and `u[i]`, `s[i]`, and `v[i]` with the i-th left-singular
 * function, i-th singular value, and i-th right-singular function, respectively.
 */
int spir_sve(const spir_kernel *k,
             spir_function *u, double *s, spir_function *v, int *n,
             double eps);

/** Fill [taumin, taumax] with the domain of the function */
int spir_func_domain(const spir_function *f, double *taumin, double *taumax);

/**
 * Takes a function `f` and arrays of size `n`: `tau`, `out`.
 * On exit, set `out[n] = f(tau[n])` to double precision.
 */
int spir_tau_value(const spir_function *f, const double *tau,
                   double *out, int n);

/**
 * Takes a function `f` and a buffer `f0` of size at least `n`. On exit, set
 * `f0[i]` to the `i`-th root of `f` and set `n` to the number of roots found.
 *
 * Return 0 on success, -1 if more roots were found.
 */
int spir_tau_roots(const spir_function *f, double *f0, int *n);

/**
 * Takes a function `f` and two arrays of arrays of size `n`: `iw`, `out`.
 * On exit, set `out[n] = FT(f)(iw[n])` to double precision, where `FT` denotes
 * the Fourier transform.
 */
int spir_iw_value(const spir_function *f, const long *iw,
                  double _Complex *out, int n);

/**
 * Takes a function `f` and a buffer `f0` of size at least `n`. On exit, set
 * `f0[i]` to the `i`-th sign change of the Fourier transform of `f` and set
 * `n` to the number of roots found.
 *
 * Return 0 on success, -1 if more roots were found.
 */
int spir_iw_roots(const spir_function *f, long *f0, int n);

#ifdef __cplusplus
}
#endif
