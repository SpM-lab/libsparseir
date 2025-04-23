#pragma once

#include <stdbool.h>
#include <complex.h>
#include <stdint.h>

#include "spir_status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Define a C-compatible type alias for the C99 complex number.
typedef double _Complex c_complex;

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

//DECLARE_OPAQUE_TYPE(polyvector);

DECLARE_OPAQUE_TYPE(singular_funcs);
DECLARE_OPAQUE_TYPE(matsubara_functions);

DECLARE_OPAQUE_TYPE(fermionic_finite_temp_basis);
DECLARE_OPAQUE_TYPE(bosonic_finite_temp_basis);
DECLARE_OPAQUE_TYPE(sampling);
DECLARE_OPAQUE_TYPE(sve_result);
DECLARE_OPAQUE_TYPE(fermionic_dlr);
DECLARE_OPAQUE_TYPE(bosonic_dlr);

/**
 * @brief Creates a new logistic kernel for fermionic/bosonic analytical
 * continuation.
 *
 * In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is
 * a function on [-1, 1] × [-1, 1]:
 *
 * K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))
 *
 * While LogisticKernel is primarily a fermionic analytic continuation kernel,
 * it can also model the τ dependence of a bosonic correlation function as:
 *
 * ∫ [exp(-Λy(x + 1)/2)/(1 - exp(-Λy))] ρ(y) dy = ∫ K(x, y) ρ'(y) dy
 *
 * where ρ'(y) = w(y)ρ(y) and the weight function w(y) = 1/tanh(Λy/2)
 *
 * @param lambda The cutoff parameter Λ (must be non-negative)
 * @return A pointer to the newly created kernel object, or NULL if creation
 * fails
 */
spir_kernel *spir_logistic_kernel_new(double lambda);

/**
 * @brief Creates a new regularized bosonic kernel for analytical continuation.
 *
 * In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is
 * a function on [-1, 1] × [-1, 1]:
 *
 * K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1)
 *
 * Special care is taken in evaluating this expression around y = 0 to handle
 * the singularity. The kernel is specifically designed for bosonic functions
 * and includes proper regularization to handle numerical stability issues.
 *
 * @param lambda The cutoff parameter Λ (must be non-negative)
 * @return A pointer to the newly created kernel object, or NULL if creation
 * fails
 *
 * @note This kernel is specifically designed for bosonic correlation functions
 *       and should not be used for fermionic cases.
 */
spir_kernel *spir_regularized_bose_kernel_new(double lambda);

/**
 * @brief Perform truncated singular value expansion (SVE) of a kernel.
 *
 * Computes a truncated singular value expansion of an integral kernel
 * K: [xmin, xmax] × [ymin, ymax] → ℝ in the form:
 *
 * K(x, y) = ∑ s[l] * u[l](x) * v[l](y)  for l = 1, 2, 3, ...
 *
 * where:
 * - s[l] are singular values in non-increasing order
 * - u[l](x) are left singular functions, forming an orthonormal system on
 * [xmin, xmax]
 * - v[l](y) are right singular functions, forming an orthonormal system on
 * [ymin, ymax]
 *
 * The SVE is computed by mapping it onto a singular value decomposition (SVD)
 * of a matrix using piecewise Legendre polynomial expansion.
 *
 * @param k Pointer to the kernel object for which to compute SVE
 * @param epsilon Accuracy target for the basis. Determines:
 *               - The relative magnitude of included singular values
 *               - The accuracy of computed singular values and vectors
 *
 * @return A pointer to the newly created SVE result object containing the
 * truncated singular value expansion, or NULL if creation fails
 *
 * @note The computation automatically uses optimized strategies:
 *       - For centrosymmetric kernels, specialized algorithms are employed
 *       - The working precision is adjusted to meet accuracy requirements
 *
 * @note The returned object must be freed using spir_destroy_sve_result when no
 * longer needed
 * @see spir_destroy_sve_result
 */
spir_sve_result* spir_sve_result_new(const spir_kernel* k, double epsilon);


/**
 * @brief Retrieves the domain boundaries of a kernel function.
 *
 * This function obtains the domain boundaries (ranges) for both the x and y
 * variables of the specified kernel function. The kernel domain is typically
 * defined as a rectangle in the (x,y) plane.
 *
 * @param k Pointer to the kernel object whose domain is to be retrieved
 * @param xmin Pointer to store the minimum value of the x-range
 * @param xmax Pointer to store the maximum value of the x-range
 * @param ymin Pointer to store the minimum value of the y-range
 * @param ymax Pointer to store the maximum value of the y-range
 *
 * @return 0 on success, -1 on failure (if the kernel is invalid or an exception
 * occurs)
 */
int spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                       double *ymin, double *ymax);

/**
 * Takes a kernel `k`, an array `x` of size `nx`, an array `y` of size `ny`
 * and an array `out` of size `nx * ny`. On exit, set
 * `out[ix*ny + iy] = K(x[ix], y[iy])`.
 */
int spir_kernel_matrix(const spir_kernel *k, const double *x, int nx,
                       const double *y, int ny, double *out);

int spir_fermionic_finite_temp_basis_get_size(const spir_fermionic_finite_temp_basis *b, int *size);

int spir_bosonic_finite_temp_basis_get_size(const spir_bosonic_finite_temp_basis *b, int *size);


/**
 * @brief Creates a new fermionic tau sampling object for sparse sampling in
 * imaginary time.
 *
 * Constructs a sampling object that allows transformation between the IR basis
 * and a set of sampling points in imaginary time (τ). The sampling points are
 * automatically chosen as the extrema of the highest-order basis function in
 * imaginary time, which provides near-optimal conditioning for the given basis
 * size.
 *
 * @param b Pointer to a fermionic finite temperature basis object
 * @return A pointer to the newly created sampling object, or NULL if creation
 * fails
 *
 * @note The sampling points are chosen to optimize numerical stability and
 * accuracy
 * @note The sampling matrix is automatically factorized using SVD for efficient
 * transformations
 * @note The returned object must be freed using spir_destroy_sampling when no
 * longer needed
 * @see spir_destroy_sampling
 */
spir_sampling *spir_fermionic_tau_sampling_new(const spir_fermionic_finite_temp_basis *b);

/**
 * @brief Creates a new fermionic Matsubara sampling object for sparse sampling
 * in Matsubara frequencies.
 *
 * Constructs a sampling object that allows transformation between the IR basis
 * and a set of sampling points in Matsubara frequencies (iωn). The sampling
 * points are automatically chosen as the (discrete) extrema of the
 * highest-order basis function in Matsubara frequencies, which provides
 * near-optimal conditioning for the given basis size.
 *
 * For fermionic Matsubara frequencies, the sampling points are odd integers:
 * iωn = (2n + 1)π/β, where n is an integer.
 *
 * @param b Pointer to a fermionic finite temperature basis object
 * @return A pointer to the newly created sampling object, or NULL if creation
 * fails
 *
 * @note The sampling points are chosen to optimize numerical stability and
 * accuracy
 * @note The sampling matrix is automatically factorized using SVD for efficient
 * transformations
 * @note For fermionic functions, the Matsubara frequencies are odd multiples of
 * π/β, i.e. iωn = (2n + 1)π/β.
 * @note The returned object must be freed using spir_destroy_sampling when no
 * longer needed
 * @see spir_destroy_sampling
 */
spir_sampling *spir_fermionic_matsubara_sampling_new(const spir_fermionic_finite_temp_basis *b);


/**
 * @brief Creates a new bosonic tau sampling object for sparse sampling in
 * imaginary time.
 *
 * Constructs a sampling object that allows transformation between the IR basis
 * and a set of sampling points in imaginary time (τ). The sampling points are
 * automatically chosen as the extrema of the highest-order basis function in
 * imaginary time, which provides near-optimal conditioning for the given basis
 * size.
 *
 * @param b Pointer to a bosonic finite temperature basis object
 * @return A pointer to the newly created sampling object, or NULL if creation
 * fails
 *
 * @note The sampling points are chosen to optimize numerical stability and
 * accuracy
 * @note The sampling matrix is automatically factorized using SVD for efficient
 * transformations
 * @note The returned object must be freed using spir_destroy_sampling when no
 * longer needed
 * @see spir_destroy_sampling
 */
spir_sampling *spir_bosonic_tau_sampling_new(const spir_bosonic_finite_temp_basis *b);

/**
 * @brief Creates a new bosonic Matsubara sampling object for sparse sampling
 * in Matsubara frequencies.
 *
 * Constructs a sampling object that allows transformation between the IR basis
 * and a set of sampling points in Matsubara frequencies (iωn). The sampling
 * points are automatically chosen as the (discrete) extrema of the
 * highest-order basis function in Matsubara frequencies, which provides
 * near-optimal conditioning for the given basis size.
 *
 * For bosonic Matsubara frequencies, the sampling points are even integers:
 * iωn = 2nπ/β, where n is an integer.
 *
 * @param b Pointer to a bosonic finite temperature basis object
 * @return A pointer to the newly created sampling object, or NULL if creation
 * fails
 *
 * @note The sampling points are chosen to optimize numerical stability and
 * accuracy
 * @note The sampling matrix is automatically factorized using SVD for efficient
 * transformations
 * @note For bosonic functions, the Matsubara frequencies are even multiples of
 * π/β
 * @note The returned object must be freed using spir_destroy_sampling when no
 * longer needed
 * @see spir_destroy_sampling
 */
spir_sampling *spir_bosonic_matsubara_sampling_new(const spir_bosonic_finite_temp_basis *b);


/**
 * @brief Creates a new fermionic Discrete Lehmann Representation (DLR).
 *
 * This function implements a variant of the discrete Lehmann representation
 * (DLR). Unlike the IR which uses truncated singular value expansion of the
 * analytic continuation kernel K, the DLR is based on a "sketching" of K. The
 * resulting basis is a linear combination of discrete set of poles on the
 * real-frequency axis, continued to the imaginary-frequency axis:
 *
 * G(iν) = ∑ a[i] / (iν - w[i]) for i = 1, 2, ..., L
 *
 * where:
 * - a[i] are the expansion coefficients
 * - w[i] are the poles on the real axis
 * - iν are the fermionic Matsubara frequencies
 *
 * @param b Pointer to a fermionic finite temperature basis object
 * @return A pointer to the newly created DLR object, or NULL if creation fails
 *
 * @note The poles on the real-frequency axis are selected based on the zeros of
 *       the IR basis functions on the real axis
 * @note The returned object must be freed using spir_destroy_fermionic_dlr when
 * no longer needed
 * @see spir_destroy_fermionic_dlr
 * @see spir_fermionic_dlr_new_with_poles
 *
 * @warning This implementation uses a heuristic approach for pole selection,
 * which differs from the original DLR method that uses rank-revealing
 * decomposition
 */
spir_fermionic_dlr *spir_fermionic_dlr_new(const spir_fermionic_finite_temp_basis *b);

/**
 * @brief Creates a new fermionic Discrete Lehmann Representation (DLR) with
 * custom poles.
 *
 * This function creates a fermionic DLR using a set of user-specified poles on
 * the real-frequency axis. The DLR represents Green's functions as a sum of
 * poles:
 *
 * G(iν) = ∑ a[i] / (iν - w[i]) for i = 1, 2, ..., npoles
 *
 * where w[i] are the specified poles and a[i] are the expansion coefficients.
 *
 * @param b Pointer to a fermionic finite temperature basis object
 * @param npoles Number of poles to use in the representation
 * @param poles Array of pole locations on the real-frequency axis
 *
 * @return A pointer to the newly created DLR object with custom poles, or NULL
 * if creation fails
 *
 * @note This function allows for more control over the pole selection compared
 * to the automatic pole selection in spir_fermionic_dlr_new
 * @see spir_fermionic_dlr_new
 * @see spir_destroy_fermionic_dlr
 */
spir_fermionic_dlr *spir_fermionic_dlr_new_with_poles(const spir_fermionic_finite_temp_basis *b, const int npoles, const double *poles);

/**
 * @brief Creates a new bosonic Discrete Lehmann Representation (DLR).
 *
 * This function implements a variant of the discrete Lehmann representation
 * (DLR). Unlike the IR which uses truncated singular value expansion of the
 * analytic continuation kernel K, the DLR is based on a "sketching" of K. The
 * resulting basis is a linear combination of discrete set of poles on the
 * real-frequency axis, continued to the imaginary-frequency axis:
 *
 * G(iωn) = ∑ a[i] / (iωn - w[i]) for i = 1, 2, ..., L
 *
 * where:
 * - a[i] are the expansion coefficients
 * - w[i] are the poles on the real axis
 * - iωn are the bosonic Matsubara frequencies (even multiples of π/β)
 *
 * @param b Pointer to a bosonic finite temperature basis object
 * @return A pointer to the newly created DLR object, or NULL if creation fails
 *
 * @note The poles on the real-frequency axis are selected based on the zeros of
 *       the IR basis functions on the real axis
 * @note The returned object must be freed using spir_destroy_bosonic_dlr when
 * no longer needed
 * @see spir_destroy_bosonic_dlr
 * @see spir_bosonic_dlr_new_with_poles
 *
 * @warning This implementation uses a heuristic approach for pole selection,
 * which differs from the original DLR method that uses rank-revealing
 * decomposition
 */
spir_bosonic_dlr *spir_bosonic_dlr_new(const spir_bosonic_finite_temp_basis *b);

/**
 * @brief Creates a new bosonic Discrete Lehmann Representation (DLR) with
 * custom poles.
 *
 * This function creates a bosonic DLR using a set of user-specified poles on
 * the real-frequency axis. The DLR represents correlation functions as a sum of
 * poles:
 *
 * G(iωn) = ∑ a[i] / (iωn - w[i]) for i = 1, 2, ..., npoles
 *
 * where w[i] are the specified poles and a[i] are the expansion coefficients.
 *
 * @param b Pointer to a bosonic finite temperature basis object
 * @param npoles Number of poles to use in the representation
 * @param poles Array of pole locations on the real-frequency axis
 *
 * @return A pointer to the newly created DLR object with custom poles, or NULL
 * if creation fails
 *
 * @note This function allows for more control over the pole selection compared
 * to the automatic pole selection in spir_bosonic_dlr_new
 * @see spir_bosonic_dlr_new
 * @see spir_destroy_bosonic_dlr
 */
spir_bosonic_dlr *spir_bosonic_dlr_new_with_poles(const spir_bosonic_finite_temp_basis *b, const int npoles, const double *poles);

/**
 * @brief Evaluates basis coefficients at sampling points (double to double
 * version).
 *
 * Transforms basis coefficients to values at sampling points, where both input
 * and output are real (double precision) values. The operation can be performed
 * along any dimension of a multidimensional array.
 *
 * @param s Pointer to the sampling object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of basis coefficients
 * @param out Output array for the evaluated values at sampling points
 *
 * @return 0 on success, non-zero on failure
 *
 * @note For optimal performance, the target dimension should be either the
 * first (0) or the last (ndim-1) dimension to avoid large temporary array
 * allocations
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 *
 * @see spir_sampling_evaluate_dz
 * @see spir_sampling_evaluate_zz
 */
int spir_sampling_evaluate_dd(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const double *input,          // Input coefficients array
    double *out                    // Output array
    );

/**
 * @brief Evaluates basis coefficients at sampling points (double to complex
 * version).
 *
 * Transforms basis coefficients to values at sampling points, where input is
 * real (double precision) and output is complex (double precision) values. The
 * operation can be performed along any dimension of a multidimensional array.
 *
 * @param s Pointer to the sampling object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of real basis coefficients
 * @param out Output array for the evaluated complex values at sampling points
 *
 * @return 0 on success, non-zero on failure
 *
 * @note For optimal performance, the target dimension should be either the
 * first (0) or the last (ndim-1) dimension to avoid large temporary array
 * allocations
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note Complex numbers are stored as pairs of consecutive double values (real,
 * imag)
 *
 * @see spir_sampling_evaluate_dd
 * @see spir_sampling_evaluate_zz
 */
int spir_sampling_evaluate_dz(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const double *input,          // Input coefficients array
    c_complex *out                    // Output array
    );

int spir_sampling_evaluate_zz(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const c_complex *input,          // Input coefficients array
    c_complex *out                    // Output array
    );


/**
 * @brief Fits values at sampling points to basis coefficients (double to double version).
 *
 * Transforms values at sampling points back to basis coefficients, where both input
 * and output are real (double precision) values. The operation can be performed
 * along any dimension of a multidimensional array.
 *
 * @param s Pointer to the sampling object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of values at sampling points
 * @param out Output array for the fitted basis coefficients
 *
 * @return SPIR_COMPUTATION_SUCCESS on success, non-zero on failure
 *
 * @note For optimal performance, the target dimension should be either the first (0)
 *       or the last (ndim-1) dimension to avoid large temporary array allocations
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note This function performs the inverse operation of spir_sampling_evaluate_dd
 *
 * @see spir_sampling_evaluate_dd
 * @see spir_sampling_fit_zz
 */
int spir_sampling_fit_dd(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const double *input,          // Input coefficients array
    double *out                    // Output array
    );

/**
 * @brief Fits values at sampling points to basis coefficients (complex to complex version).
 *
 * Transforms values at sampling points back to basis coefficients, where both input
 * and output are complex (double precision) values. The operation can be performed
 * along any dimension of a multidimensional array.
 *
 * @param s Pointer to the sampling object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of complex values at sampling points
 * @param out Output array for the fitted complex basis coefficients
 *
 * @return SPIR_COMPUTATION_SUCCESS on success, non-zero on failure
 *
 * @note For optimal performance, the target dimension should be either the first (0)
 *       or the last (ndim-1) dimension to avoid large temporary array allocations
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note Complex numbers are stored as pairs of consecutive double values (real, imag)
 * @note This function performs the inverse operation of spir_sampling_evaluate_zz
 *
 * @see spir_sampling_evaluate_zz
 * @see spir_sampling_fit_dd
 */
int spir_sampling_fit_zz(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const c_complex *input,          // Input coefficients array
    c_complex *out                    // Output array
    );

/**
 * @brief Gets the number of rows in the fitting matrix of a bosonic DLR.
 *
 * This function returns the number of rows in the fitting matrix of the specified
 * bosonic Discrete Lehmann Representation (DLR). The fitting matrix is used to
 * transform between the DLR representation and values at sampling points.
 *
 * @param dlr Pointer to the bosonic DLR object
 * @return The number of rows in the fitting matrix, or SPIR_GET_IMPL_FAILED if the DLR object is invalid
 *
 * @note The fitting matrix dimensions determine the size of valid input/output arrays
 *       for transformations involving this DLR object
 * @see spir_bosonic_dlr_fitmat_cols
 * @see spir_bosonic_dlr_from_IR
 * @see spir_bosonic_dlr_to_IR
 */
int spir_bosonic_dlr_fitmat_rows(const spir_bosonic_dlr *dlr);

/**
 * @brief Gets the number of columns in the fitting matrix of a bosonic DLR.
 *
 * This function returns the number of columns in the fitting matrix of the specified
 * bosonic Discrete Lehmann Representation (DLR). The fitting matrix is used to
 * transform between the DLR representation and values at sampling points.
 *
 * @param dlr Pointer to the bosonic DLR object
 * @return The number of columns in the fitting matrix, or SPIR_GET_IMPL_FAILED if the DLR object is invalid
 *
 * @note The fitting matrix dimensions determine the size of valid input/output arrays
 *       for transformations involving this DLR object
 * @see spir_bosonic_dlr_fitmat_rows
 * @see spir_bosonic_dlr_from_IR
 * @see spir_bosonic_dlr_to_IR
 */
int spir_bosonic_dlr_fitmat_cols(const spir_bosonic_dlr *dlr);

/**
 * @brief Gets the number of rows in the fitting matrix of a fermionic DLR.
 *
 * This function returns the number of rows in the fitting matrix of the specified
 * fermionic Discrete Lehmann Representation (DLR). The fitting matrix is used to
 * transform between the DLR representation and values at sampling points.
 *
 * @param dlr Pointer to the fermionic DLR object
 * @return The number of rows in the fitting matrix, or SPIR_GET_IMPL_FAILED if the DLR object is invalid
 *
 * @note The fitting matrix dimensions determine the size of valid input/output arrays
 *       for transformations involving this DLR object
 * @see spir_fermionic_dlr_fitmat_cols
 * @see spir_fermionic_dlr_from_IR
 * @see spir_fermionic_dlr_to_IR
 */
int spir_fermionic_dlr_fitmat_rows(const spir_fermionic_dlr *dlr);
/**
 * @brief Gets the number of columns in the fitting matrix of a fermionic DLR.
 *
 * This function returns the number of columns in the fitting matrix of the specified
 * fermionic Discrete Lehmann Representation (DLR). The fitting matrix is used to
 * transform between the DLR representation and values at sampling points.
 *
 * @param dlr Pointer to the fermionic DLR object
 * @return The number of columns in the fitting matrix, or SPIR_GET_IMPL_FAILED if the DLR object is invalid
 *
 * @note The fitting matrix dimensions determine the size of valid input/output arrays
 *       for transformations involving this DLR object
 * @see spir_fermionic_dlr_fitmat_rows
 * @see spir_fermionic_dlr_from_IR
 * @see spir_fermionic_dlr_to_IR
 */
int spir_fermionic_dlr_fitmat_cols(const spir_fermionic_dlr *dlr);


/**
 * Transforms a given input array from the Imaginary Frequency (IR) representation
 * to the Fermionic Discrete Lehmann Representation (DLR) using the specified DLR object.
 *
 * @param dlr Pointer to the fermionic DLR object
 * @param order Order type (C or Fortran)
 * @param ndim Number of dimensions
 * @param input_dims Array of dimensions
 * @param input Input coefficients array in IR representation
 * @param out Output array in DLR representation
 *
 * @return 0 on success, or a negative value if an error occurred
 *
 * @note The input and output arrays must be allocated with sufficient memory.
 *       The size of the input and output arrays should match the dimensions specified.
 *       The order type determines the memory layout of the input and output arrays.
 *       The function assumes that the input array is in the specified order type.
 *       The output array will be in the specified order type.
 *
 * @see spir_fermionic_dlr_to_IR
 * @see spir_fermionic_dlr_fitmat_rows
 * @see spir_fermionic_dlr_fitmat_cols
 */
int spir_fermionic_dlr_from_IR(
    const spir_fermionic_dlr *dlr,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    const double *input,
    double *out);/**
 * @brief Transforms coefficients from IR basis to bosonic DLR representation.
 *
 * This function converts expansion coefficients from the Intermediate
 * Representation (IR) basis to the Discrete Lehmann Representation (DLR).
 * The transformation is performed by solving a linear system using the
 * fitting matrix:
 *
 * g_DLR = matrix \ g_IR
 *
 * where:
 * - g_DLR are the coefficients in the DLR basis
 * - g_IR are the coefficients in the IR basis
 * - matrix is the SVD-factorized transformation matrix
 *
 * @param dlr Pointer to the bosonic DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param input Input array of IR coefficients (double precision)
 * @param out Output array for the DLR coefficients (double precision)
 *
 * @return SPIR_COMPUTATION_SUCCESS on success, SPIR_GET_IMPL_FAILED on failure
 * (if the DLR object is invalid or an error occurs)
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note This function is specifically for bosonic (symmetric) Green's functions
 * @note The transformation preserves the numerical properties of the
 * representation
 * @note The transformation involves solving a linear system, which may be
 *       computationally more intensive than the forward transformation
 *
 * @see spir_bosonic_dlr_to_IR
 * @see spir_fermionic_dlr_from_IR
 */
int spir_bosonic_dlr_from_IR(
    const spir_bosonic_dlr *dlr,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    const double *input,
    double *out);

/**
 * @brief Transforms coefficients from DLR basis to fermionic IR representation.
 *
 * This function converts expansion coefficients from the Discrete Lehmann
 * Representation (DLR) basis to the Intermediate Representation (IR) basis.
 * The transformation is performed using the fitting matrix:
 *
 * g_IR = fitmat * g_DLR
 *
 * where:
 * - g_IR are the coefficients in the IR basis
 * - g_DLR are the coefficients in the DLR basis
 * - fitmat is the transformation matrix
 *
 * @param dlr Pointer to the fermionic DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param input Input array of DLR coefficients (double precision)
 * @param out Output array for the IR coefficients (double precision)
 *
 * @return SPIR_COMPUTATION_SUCCESS on success, SPIR_GET_IMPL_FAILED on failure (if the DLR object is invalid or an error
 * occurs)
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note This function is specifically for fermionic Green's functions
 * @note The transformation is a direct matrix multiplication, which is
 * typically faster than the inverse transformation
 *
 * @see spir_fermionic_dlr_from_IR
 * @see spir_bosonic_dlr_to_IR
 */
int spir_fermionic_dlr_from_IR(
    const spir_fermionic_dlr *dlr,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    const double *input,
    double *out);

/**
 * @brief Transforms coefficients from DLR basis to bosonic IR representation.
 *
 * This function converts expansion coefficients from the Discrete Lehmann
 * Representation (DLR) basis to the Intermediate Representation (IR) basis.
 * The transformation is performed using the fitting matrix:
 *
 * g_IR = fitmat * g_DLR
 *
 * where:
 * - g_IR are the coefficients in the IR basis
 * - g_DLR are the coefficients in the DLR basis
 * - fitmat is the transformation matrix
 *
 * @param dlr Pointer to the bosonic DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param input Input array of DLR coefficients (double precision)
 * @param out Output array for the IR coefficients (double precision)
 *
 * @return SPIR_COMPUTATION_SUCCESS on success, SPIR_GET_IMPL_FAILED on failure
 *         (if the DLR object is invalid or an error occurs)
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note This function is specifically for bosonic (symmetric) Green's functions
 * @note The transformation is a direct matrix multiplication, which is
 *       typically faster than the inverse transformation
 *
 * @see spir_bosonic_dlr_from_IR
 * @see spir_fermionic_dlr_to_IR
 */
int spir_bosonic_dlr_to_IR(
    const spir_bosonic_dlr *dlr,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    const double *input,
    double *out);

/**
 * @brief Transforms coefficients from DLR basis to fermionic IR representation.
 *
 * This function converts expansion coefficients from the Discrete Lehmann
 * Representation (DLR) basis to the Intermediate Representation (IR) basis.
 * The transformation is performed using the fitting matrix:
 *
 * g_IR = fitmat * g_DLR
 *
 * where:
 * - g_IR are the coefficients in the IR basis
 * - g_DLR are the coefficients in the DLR basis
 * - fitmat is the transformation matrix
 *
 * @param dlr Pointer to the fermionic DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param input Input array of DLR coefficients (double precision)
 * @param out Output array for the IR coefficients (double precision)
 *
 * @return SPIR_COMPUTATION_SUCCESS on success, SPIR_GET_IMPL_FAILED on failure
 *         (if the DLR object is invalid or an error occurs)
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note This function is specifically for fermionic Green's functions
 * @note The transformation is a direct matrix multiplication, which is
 *       typically faster than the inverse transformation
 *
 * @see spir_fermionic_dlr_from_IR
 * @see spir_bosonic_dlr_to_IR
 */
int spir_fermionic_dlr_to_IR(
    const spir_fermionic_dlr *dlr,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    const double *input,
    double *out);

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
/**
 * @brief Creates a new bosonic finite temperature IR basis.
 *
 * For a continuation kernel K from real frequencies, ω ∈ [-ωmax, ωmax], to
 * imaginary time, τ ∈ [0, β], this function creates an intermediate
 * representation (IR) basis that stores the truncated singular value expansion:
 *
 * K(τ, ω) ≈ ∑ u[l](τ) * s[l] * v[l](ω) for l = 1, 2, 3, ...
 *
 * where:
 * - u[l](τ) are IR basis functions on the imaginary time axis (stored as
 * piecewise Legendre polynomials)
 * - s[l] are singular values of the continuation kernel
 * - v[l](ω) are IR basis functions on the real frequency axis (stored as
 * piecewise Legendre polynomials)
 *
 * @param beta Inverse temperature β (must be positive)
 * @param omega_max Frequency cutoff ωmax (must be non-negative)
 * @param epsilon Accuracy target for the basis
 *
 * @return A pointer to the newly created bosonic finite temperature basis
 * object, or NULL if creation fails
 *
 * @note The basis includes both imaginary time and Matsubara frequency
 * representations
 * @note For Matsubara frequencies, bosonic basis uses even numbers (2n)
 * @note The returned object must be freed using
 * spir_destroy_bosonic_finite_temp_basis when no longer needed
 * @see spir_destroy_bosonic_finite_temp_basis
 */
spir_bosonic_finite_temp_basis* spir_bosonic_finite_temp_basis_new(double beta, double omega_max, double epsilon);

/**
 * @brief Creates a new fermionic finite temperature IR basis using a
 * pre-computed SVE result.
 *
 * This function creates a fermionic intermediate representation (IR) basis
 * using a pre-computed singular value expansion (SVE) result. This allows for
 * reusing an existing SVE computation, which can be more efficient than
 * recomputing it.
 *
 * @param beta Inverse temperature β (must be positive)
 * @param omega_max Frequency cutoff ωmax (must be non-negative)
 * @param k Pointer to the kernel object used for the basis construction
 * @param sve Pointer to a pre-computed SVE result for the kernel
 *
 * @return A pointer to the newly created fermionic finite temperature basis
 * object, or NULL if creation fails (invalid inputs or exception occurs)
 *
 * @note Using a pre-computed SVE can significantly improve performance when
 * creating multiple basis objects with the same kernel
 * @see spir_sve_result_new
 * @see spir_destroy_fermionic_finite_temp_basis
 */
spir_fermionic_finite_temp_basis *
spir_fermionic_finite_temp_basis_new_with_sve(double beta, double omega_max,
                                             const spir_kernel *k,
                                             const spir_sve_result *sve);

/**
 * @brief Creates a new bosonic finite temperature IR basis using a pre-computed
 * SVE result.
 *
 * This function creates a bosonic intermediate representation (IR) basis using
 * a pre-computed singular value expansion (SVE) result. This allows for reusing
 * an existing SVE computation, which can be more efficient than recomputing it.
 *
 * @param beta Inverse temperature β (must be positive)
 * @param omega_max Frequency cutoff ωmax (must be non-negative)
 * @param k Pointer to the kernel object used for the basis construction
 * @param sve Pointer to a pre-computed SVE result for the kernel
 *
 * @return A pointer to the newly created bosonic finite temperature basis
 * object, or NULL if creation fails (invalid inputs or exception occurs)
 *
 * @note Using a pre-computed SVE can significantly improve performance when
 * creating multiple basis objects with the same kernel
 * @see spir_sve_result_new
 * @see spir_destroy_bosonic_finite_temp_basis
 */
spir_bosonic_finite_temp_basis *
spir_bosonic_finite_temp_basis_new_with_sve(double beta, double omega_max,
                                             const spir_kernel *k,
                                             const spir_sve_result *sve);

// Destroy basis instance
//void spir_destroy_fermionic_basis(spir_fermionic_finite_temp_basis* b);

/**
 * @brief Gets the basis functions of a fermionic finite temperature basis.
 *
 * This function returns a polynomial vector containing the basis functions of the
 * specified fermionic finite temperature basis. The basis functions are stored
 * as piecewise Legendre polynomials.
 *
 * @param b Pointer to the fermionic finite temperature basis object
 * @return A pointer to the polynomial vector containing the basis functions,
 *         or NULL if the basis object is invalid
 *
 * @note The returned polynomial vector must be freed using spir_destroy_polyvector
 *       when no longer needed
 * @see spir_destroy_polyvector
 */
spir_singular_funcs* spir_fermionic_finite_temp_basis_get_u(const spir_fermionic_finite_temp_basis* b);


/**
 * @brief Gets the basis functions of a fermionic finite temperature basis.
 *
 * This function returns a polynomial vector containing the basis functions of the
 * specified fermionic finite temperature basis. The basis functions are stored
 * as piecewise Legendre polynomials.
 *
 * @param b Pointer to the fermionic finite temperature basis object
 * @return A pointer to the polynomial vector containing the basis functions,
 *         or NULL if the basis object is invalid
 *
 * @note The returned polynomial vector must be freed using spir_destroy_polyvector
 *       when no longer needed
 * @see spir_destroy_polyvector
 */
spir_singular_funcs* spir_fermionic_finite_temp_basis_get_v(const spir_fermionic_finite_temp_basis* b);

/**
 * @brief Gets the basis functions of a fermionic finite temperature basis in Matsubara frequency domain.
 *
 * This function returns a polynomial vector containing the basis functions of the
 * specified fermionic finite temperature basis in Matsubara frequency domain.
 *
 * @param b Pointer to the fermionic finite temperature basis object
 * @return A pointer to the object containing the basis functions,
 *         or NULL if the basis object is invalid
 *
 * @note The returned object must be freed using spir_destroy_matsubara_functions
 *       when no longer needed
 * @see spir_destroy_matsubara_functions
 */
spir_matsubara_functions* spir_fermionic_finite_temp_basis_get_uhat(const spir_fermionic_finite_temp_basis* b);

/**
 * @brief Gets the basis functions of a bosonic finite temperature basis.
 *
 * This function returns a polynomial vector containing the basis functions of the
 * specified bosonic finite temperature basis. The basis functions are stored
 * as piecewise Legendre polynomials.
 *
 * @param b Pointer to the bosonic finite temperature basis object
 * @return A pointer to the polynomial vector containing the basis functions,
 *         or NULL if the basis object is invalid
 *
 * @note The returned polynomial vector must be freed using spir_destroy_polyvector
 *       when no longer needed
 * @see spir_destroy_polyvector
 */
spir_singular_funcs* spir_bosonic_finite_temp_basis_get_u(const spir_bosonic_finite_temp_basis* b);

/**
 * @brief Gets the basis functions of a bosonic finite temperature basis on the real frequency axis.
 *
 * This function returns a polynomial vector containing the basis functions of the
 * specified bosonic finite temperature basis on the real frequency axis. The basis functions are stored
 * as piecewise Legendre polynomials.
 *
 * @param b Pointer to the bosonic finite temperature basis object
 * @return A pointer to the polynomial vector containing the basis functions,
 *         or NULL if the basis object is invalid
 *
 * @note The returned polynomial vector must be freed using spir_destroy_polyvector
 *       when no longer needed
 * @see spir_destroy_polyvector
 */
spir_singular_funcs* spir_bosonic_finite_temp_basis_get_v(const spir_bosonic_finite_temp_basis* b);

/**
 * @brief Gets the basis functions of a bosonic finite temperature basis in Matsubara frequency domain.
 *
 * This function returns a polynomial vector containing the basis functions of the
 * specified bosonic finite temperature basis in Matsubara frequency domain.
 *
 * @param b Pointer to the bosonic finite temperature basis object
 * @return A pointer to the object containing the basis functions,
 *         or NULL if the basis object is invalid
 *
 * @note The returned object must be freed using spir_destroy_matsubara_functions
 *       when no longer needed
 * @see spir_destroy_matsubara_functions
 */
spir_matsubara_functions* spir_bosonic_finite_temp_basis_get_uhat(const spir_bosonic_finite_temp_basis* b);

/**
 * @brief Evaluates basis functions at a single point in the imaginary-time domain or the real frequency domain.
 *
 * This function evaluates all basis functions contained in a polynomial vector at a specified point x.
 * The values of each basis function at x are stored in the output array.
 * The output array out[j] contains the value of the j-th basis function evaluated at x.
 *
 * @param uv Pointer to the polynomial vector containing the basis functions
 * @param x Point at which to evaluate the basis functions
 * @param out Pre-allocated array to store the evaluation results. Must have size >= n_basis
 * @return SPIR_COMPUTATION_SUCCESS on success, error code on failure
 *
 * @note The output array must be pre-allocated with sufficient size to store all basis function values
 */
int32_t spir_evaluate_singular_funcs(const spir_singular_funcs* uv, double x, double* out);


/**
 * @brief Evaluates basis functions at multiple Matsubara frequencies.
 *
 * This function evaluates all basis functions contained in a Matsubara basis functions object
 * at the specified Matsubara frequency indices. The values of each basis function at each
 * frequency are stored in the output array.
 *
 * @param uiw Pointer to the Matsubara basis functions object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * @param num_freqs Number of Matsubara frequencies at which to evaluate
 * @param matsubara_freq_indices Array of Matsubara frequency indices
 * @param out Pre-allocated array to store the evaluation results. The results are stored as a 2D array of size n_basis x num_freqs.
 * @return SPIR_COMPUTATION_SUCCESS on success, error code on failure
 *
 * @note The output array must be pre-allocated with sufficient size to store all basis function values
 *       at all requested frequencies. Indices n correspond to ωn = nπ/β,
 *       where n are odd for fermionic frequencies and even for bosonic frequencies.
 */
int32_t spir_evaluate_matsubara_functions(
    const spir_matsubara_functions* uiw,
    spir_order_type order, 
    int32_t num_freqs,
    int32_t* matsubara_freq_indices, c_complex* out);

/**
 * @brief Gets the number of sampling points in a sampling object.
 *
 * This function returns the number of sampling points used in the specified
 * sampling object. This number is needed to allocate arrays of the correct size
 * when retrieving the actual sampling points.
 *
 * @param s Pointer to the sampling object
 * @param num_points Pointer to store the number of sampling points
 * @return SPIR_COMPUTATION_SUCCESS on success, SPIR_GET_IMPL_FAILED if the sampling object is invalid
 *
 * @see spir_sampling_get_tau_points
 * @see spir_sampling_get_matsubara_points
 */
int spir_sampling_get_num_points(const spir_sampling *s, int *num_points);

/**
 * @brief Gets the imaginary time sampling points.
 *
 * This function fills the provided array with the imaginary time (τ) sampling points
 * used in the specified sampling object. The array must be pre-allocated with
 * sufficient size (use spir_sampling_get_num_points to determine the required size).
 *
 * @param s Pointer to the sampling object
 * @param points Pre-allocated array to store the τ sampling points
 * @return SPIR_COMPUTATION_SUCCESS on success
 *         SPIR_GET_IMPL_FAILED if s is invalid
 *         SPIR_NOT_SUPPORTED if the sampling object is not for τ sampling
 *
 * @note The array must be pre-allocated with size >= spir_sampling_get_num_points(s)
 * @see spir_sampling_get_num_points
 */
int spir_sampling_get_tau_points(const spir_sampling *s, double *points);

/**
 * @brief Gets the Matsubara frequency sampling points.
 *
 * This function fills the provided array with the Matsubara frequency indices (n)
 * used in the specified sampling object. The actual Matsubara frequencies are
 * ωn = (2n + 1)π/β for fermionic case and ωn = 2nπ/β for bosonic case.
 * The array must be pre-allocated with sufficient size
 * (use spir_sampling_get_num_points to determine the required size).
 *
 * @param s Pointer to the sampling object
 * @param points Pre-allocated array to store the Matsubara frequency indices
 * @return SPIR_COMPUTATION_SUCCESS on success
 *         SPIR_GET_IMPL_FAILED if s is invalid
 *         SPIR_NOT_SUPPORTED if the sampling object is not for Matsubara sampling
 *
 * @note The array must be pre-allocated with size >= spir_sampling_get_num_points(s)
 * @note For fermionic case, the indices n give frequencies ωn = (2n + 1)π/β
 * @note For bosonic case, the indices n give frequencies ωn = 2nπ/β
 * @see spir_sampling_get_num_points
 */
int spir_sampling_get_matsubara_points(const spir_sampling *s, int *points);

#ifdef __cplusplus
}
#endif
