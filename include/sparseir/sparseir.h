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
    typedef struct _spir_##name spir_##name;                                   \
                                                                               \
    /* Destroy function */                                                     \
    void spir_destroy_##name(spir_##name *obj);                                \
                                                                               \
    /* Clone function */                                                       \
    spir_##name *spir_clone_##name(const spir_##name *src);                    \
                                                                               \
    /* Check if the shared_ptr is assigned to a valid object */                \
    int spir_is_assigned_##name(const spir_##name *obj);

/* Declare opaque types */
struct _spir_kernel;

DECLARE_OPAQUE_TYPE(kernel);
DECLARE_OPAQUE_TYPE(logistic_kernel);
DECLARE_OPAQUE_TYPE(regularized_bose_kernel);
DECLARE_OPAQUE_TYPE(funcs);
DECLARE_OPAQUE_TYPE(matsubara_funcs);
DECLARE_OPAQUE_TYPE(finite_temp_basis);
DECLARE_OPAQUE_TYPE(sampling);
DECLARE_OPAQUE_TYPE(sve_result);
DECLARE_OPAQUE_TYPE(dlr);

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
 * @param kernel Pointer to a pointer that will store the newly created kernel object.
 *               On successful creation, this will point to the new kernel.
 *               If creation fails, this will be set to NULL.
 * @param lambda The cutoff parameter Λ (must be non-negative)
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on successful creation of the kernel.
 *         - A non-zero error code if creation fails (e.g., due to invalid input
 *           or memory allocation failure).
 *
 * @note The kernel is implemented using piecewise Legendre polynomial expansion
 *       for numerical stability and accuracy.
 */
int32_t spir_logistic_kernel_new(spir_kernel **kernel, double lambda);

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
 * @param kernel Pointer to a pointer that will store the newly created kernel object.
 *               On successful creation, this will point to the new kernel.
 *               If creation fails, this will be set to NULL.
 * @param lambda The cutoff parameter Λ (must be non-negative)
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on successful creation of the kernel.
 *         - A non-zero error code if creation fails (e.g., due to invalid input
 *           or memory allocation failure).
 *
 * @note This kernel is specifically designed for bosonic correlation functions
 *       and should not be used for fermionic cases.
 * @note The kernel is implemented using piecewise Legendre polynomial expansion
 *       for numerical stability and accuracy.
 */
int32_t spir_regularized_bose_kernel_new(spir_kernel **kernel, double lambda);

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
 * of a matrix using piecewise Legendre polynomial expansion. The accuracy of
 * the computation is controlled by the epsilon parameter, which determines:
 * - The relative magnitude of included singular values
 * - The accuracy of computed singular values and vectors
 *
 * @param sve Pointer to a pointer that will store the newly created SVE result.
 *            On successful creation, this will point to the new SVE result.
 *            If creation fails, this will be set to NULL.
 * @param k Pointer to the kernel object for which to compute SVE
 * @param epsilon Accuracy target for the basis. Determines:
 *               - The relative magnitude of included singular values
 *               - The accuracy of computed singular values and vectors
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on successful creation.
 *         - A non-zero error code if creation fails.
 *
 * @note The computation automatically uses optimized strategies:
 *       - For centrosymmetric kernels, specialized algorithms are employed
 *       - The working precision is adjusted to meet accuracy requirements
 *       - If epsilon is below √ε (where ε is machine epsilon), a warning is
 *         issued and higher precision arithmetic is used
 *
 * @note The returned object must be freed using spir_destroy_sve_result when no
 * longer needed
 * @see spir_destroy_sve_result
 */
int32_t spir_sve_result_new(spir_sve_result **sve, const spir_kernel *k, double epsilon);

/**
 * @brief Retrieves the domain boundaries of a kernel function.
 *
 * This function obtains the domain boundaries (ranges) for both the x and y
 * variables of the specified kernel function. The kernel domain is typically
 * defined as a rectangle in the (x,y) plane.
 *
 * @param k Pointer to the kernel object whose domain is to be retrieved.
 * @param xmin Pointer to store the minimum value of the x-range.
 * @param xmax Pointer to store the maximum value of the x-range.
 * @param ymin Pointer to store the minimum value of the y-range.
 * @param ymax Pointer to store the maximum value of the y-range.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note For the logistic and regularized bosonic kernels, the domain is
 *       typically [-1, 1] × [-1, 1] in dimensionless variables.
 */
int32_t spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                           double *ymin, double *ymax);

/**
 * @brief Gets the size (number of basis functions) of a finite temperature
 * basis.
 *
 * This function returns the number of basis functions in the specified finite
 * temperature basis object. This size determines the dimensionality of the
 * basis and is needed when allocating arrays for basis function evaluations.
 *
 * @param b Pointer to the finite temperature basis object
 * @param size Pointer to store the number of basis functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The size is determined automatically during basis construction based on
 *       the specified parameters (β, ωmax, ε) and the kernel's singular value
 *       expansion.
 */
int32_t spir_finite_temp_basis_get_size(const spir_finite_temp_basis *b,
                                        int *size);

/**
 * @brief Gets the statistics type (Fermionic or Bosonic) of a finite
 * temperature basis.
 *
 * This function returns the statistics type of the specified finite temperature
 * basis object. The statistics type determines whether the basis is for
 * fermionic or bosonic Green's functions.
 *
 * @param b Pointer to the finite temperature basis object
 * @param statistics Pointer to store the statistics type
 * (SPIR_STATISTICS_FERMIONIC or SPIR_STATISTICS_BOSONIC)
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The statistics type is determined during basis construction and cannot
 * be changed
 * @note The statistics type affects the form of the basis functions and the
 *       sampling points used for evaluation.
 */
int32_t spir_finite_temp_basis_get_statistics(const spir_finite_temp_basis *b,
                                              spir_statistics_type *statistics);

/**
 * @brief Creates a new tau sampling object for sparse sampling in
 * imaginary time.
 *
 * Constructs a sampling object that allows transformation between the IR basis
 * and a set of sampling points in imaginary time (τ). The sampling points are
 * automatically chosen as the extrema of the highest-order basis function in
 * imaginary time, which provides near-optimal conditioning for the given basis
 * size.
 *
 * @param s Pointer to a pointer that will store the newly created sampling object.
 *          On successful creation, this will point to the new sampling object.
 *          If creation fails, this will be set to NULL.
 * @param b Pointer to a finite temperature basis object
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The sampling points are chosen to optimize numerical stability and
 *       accuracy
 * @note The sampling matrix is automatically factorized using SVD for efficient
 *       transformations
 * @note The returned object must be freed using spir_destroy_sampling when no
 *       longer needed
 * @see spir_destroy_sampling
 */
int32_t spir_tau_sampling_new(spir_sampling **s, const spir_finite_temp_basis *b);

/**
 * @brief Creates a new Matsubara sampling object for sparse sampling
 * in Matsubara frequencies.
 *
 * Constructs a sampling object that allows transformation between the IR basis
 * and a set of sampling points in Matsubara frequencies (iωn). The sampling
 * points are automatically chosen as the (discrete) extrema of the
 * highest-order basis function in Matsubara frequencies, which provides
 * near-optimal conditioning for the given basis size.
 *
 * @param s Pointer to a pointer that will store the newly created sampling object.
 *          On successful creation, this will point to the new sampling object.
 *          If creation fails, this will be set to NULL.
 * @param b Pointer to a finite temperature basis object
 * @param positive_only If true, only positive Matsubara frequencies are considered.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The sampling points are chosen to optimize numerical stability and
 *       accuracy
 * @note The sampling matrix is automatically factorized using SVD for efficient
 *       transformations
 * @note The sampling frequencies are stored in the sampling object as integers,
 *       i.e, ωn = nπ/β. n are even for bosonic frequencies and odd for fermionic
 *       frequencies.
 * @note The returned object must be freed using spir_destroy_sampling when no
 *       longer needed
 * @see spir_destroy_sampling
 */
int32_t spir_matsubara_sampling_new(spir_sampling **s, const spir_finite_temp_basis *b, bool positive_only);

int32_t spir_matsubara_sampling_dlr_new(spir_sampling **s, const spir_dlr *dlr, int32_t n_smpl_points, const int32_t *smpl_points, bool positive_only);


/**
 * @brief Creates a new Discrete Lehmann Representation (DLR) basis.
 *
 * This function implements a variant of the discrete Lehmann representation
 * (DLR). Unlike the IR which uses truncated singular value expansion of the
 * analytic continuation kernel K, the DLR is based on a "sketching" of K. The
 * resulting basis is a linear combination of discrete set of poles on the
 * real-frequency axis, continued to the imaginary-frequency axis:
 *
 * G(iν) = ∑ a[i] * reg[i] / (iν - w[i]) for i = 1, 2, ..., L
 *
 * where:
 * - a[i] are the expansion coefficients
 * - w[i] are the poles on the real axis
 * - reg[i] are the regularization factors, which are 1 for fermionic
 * frequencies. For bosonic frequencies, we take reg[i] = tanh(βω[i]/2)
 * (logistic kernel), reg[i] = w[i] (regularized bosonic kernel). The DLR basis
 * functions are given by u[i](iν) = reg[i] / (iν - w[i]) in the
 * imaginary-frequency domain. In the imaginary-time domain, the basis functions
 * are given by u[i](τ) = reg[i] * exp(-w[i]τ) / (1 + exp(-w[i]β)) for fermionic
 * frequencies, u[i](τ) = reg[i] * exp(-w[i]τ) / (1 - exp(-w[i]β)) for bosonic
 * frequencies.
 * - iν are Matsubara frequencies
 *
 * @param dlr Pointer to a pointer that will store the newly created DLR object.
 *            On successful creation, this will point to the new DLR object.
 *            If creation fails, this will be set to NULL.
 * @param b Pointer to a finite temperature basis object
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The poles on the real-frequency axis are selected based on the zeros of
 *       the IR basis functions on the real axis
 * @note The returned object must be freed using spir_destroy_dlr when
 *       no longer needed
 * @see spir_destroy_dlr
 * @see spir_dlr_new_with_poles
 *
 * @warning This implementation uses a heuristic approach for pole selection,
 *          which differs from the original DLR method that uses rank-revealing
 *          decomposition
 */
int32_t spir_dlr_new(spir_dlr **dlr, const spir_finite_temp_basis *b);

/**
 * @brief Creates a new Discrete Lehmann Representation (DLR) with
 * custom poles.
 *
 * This function creates a DLR basis with user-specified pole locations on the
 * real-frequency axis. This allows for more control over the pole selection
 * compared to the automatic pole selection in spir_dlr_new.
 *
 * @param dlr Pointer to a pointer that will store the newly created DLR object.
 *            On successful creation, this will point to the new DLR object.
 *            If creation fails, this will be set to NULL.
 * @param b Pointer to a finite temperature basis object
 * @param npoles Number of poles to use in the representation
 * @param poles Array of pole locations on the real-frequency axis
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note This function allows for more control over the pole selection compared
 *       to the automatic pole selection in spir_dlr_new
 * @see spir_dlr_new
 * @see spir_destroy_dlr
 */
int32_t spir_dlr_new_with_poles(spir_dlr **dlr, const spir_finite_temp_basis *b,
                                  const int npoles, const double *poles);

/**
 * @brief Gets the statistics type of a DLR.
 *
 * This function returns the statistics type (fermionic or bosonic) of the
 * specified DLR object.
 *
 * @param dlr Pointer to the DLR object
 * @param statistics Pointer to store the statistics type
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @see spir_statistics_type
 */
int32_t spir_dlr_get_statistics(const spir_dlr *dlr,
                                spir_statistics_type *statistics);

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
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note For optimal performance, the target dimension should be either the
 *       first (0) or the last (ndim-1) dimension to avoid large temporary array
 *       allocations
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note The transformation is performed using a pre-computed sampling matrix
 *       that is factorized using SVD for efficiency
 *
 * @see spir_sampling_evaluate_dz
 * @see spir_sampling_evaluate_zz
 */
int32_t
spir_sampling_evaluate_dd(const spir_sampling *s, // Sampling object
                          spir_order_type order,  // Order type (C or Fortran)
                          int32_t ndim,           // Number of dimensions
                          int32_t *input_dims,    // Array of dimensions
                          int32_t target_dim, // Target dimension for evaluation
                          const double *input, // Input coefficients array
                          double *out          // Output array
);

/**
 * @brief Evaluates basis coefficients at sampling points (double to complex
 * version).
 *
 * For more details, see spir_sampling_evaluate_dd
 * @see spir_sampling_evaluate_dd
 */
int32_t
spir_sampling_evaluate_dz(const spir_sampling *s, // Sampling object
                          spir_order_type order,  // Order type (C or Fortran)
                          int32_t ndim,           // Number of dimensions
                          int32_t *input_dims,    // Array of dimensions
                          int32_t target_dim, // Target dimension for evaluation
                          const double *input, // Input coefficients array
                          c_complex *out       // Output array
);

/**
 * @brief Evaluates basis coefficients at sampling points (complex to complex
 * version).
 *
 * For more details, see spir_sampling_evaluate_dd
 * @see spir_sampling_evaluate_dd
 */
int32_t
spir_sampling_evaluate_zz(const spir_sampling *s, // Sampling object
                          spir_order_type order,  // Order type (C or Fortran)
                          int32_t ndim,           // Number of dimensions
                          int32_t *input_dims,    // Array of dimensions
                          int32_t target_dim, // Target dimension for evaluation
                          const c_complex *input, // Input coefficients array
                          c_complex *out          // Output array
);

/**
 * @brief Fits values at sampling points to basis coefficients (double to double
 * version).
 *
 * Transforms values at sampling points back to basis coefficients, where both
 * input and output are real (double precision) values. The operation can be
 * performed along any dimension of a multidimensional array.
 *
 * @param s Pointer to the sampling object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of values at sampling points
 * @param out Output array for the fitted basis coefficients
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with the correct size
 * @note This function performs the inverse operation of
 *       spir_sampling_evaluate_dd
 * @note The transformation is performed using a pre-computed sampling matrix
 *       that is factorized using SVD for efficiency
 *
 * @see spir_sampling_evaluate_dd
 * @see spir_sampling_fit_zz
 */
int32_t
spir_sampling_fit_dd(const spir_sampling *s, // Sampling object
                     spir_order_type order,  // Order type (C or Fortran)
                     int32_t ndim,           // Number of dimensions
                     int32_t *input_dims,    // Array of dimensions
                     int32_t target_dim,     // Target dimension for evaluation
                     const double *input,    // Input coefficients array
                     double *out             // Output array
);

/**
 * @brief Fits values at sampling points to basis coefficients (complex to
 * complex version).
 *
 * For more details, see spir_sampling_fit_dd
 * @see spir_sampling_fit_dd
 */
int spir_sampling_fit_zz(const spir_sampling *s, // Sampling object
                         spir_order_type order,  // Order type (C or Fortran)
                         int32_t ndim,           // Number of dimensions
                         int32_t *input_dims,    // Array of dimensions
                         int32_t target_dim, // Target dimension for evaluation
                         const c_complex *input, // Input coefficients array
                         c_complex *out          // Output array
);

/**
 * @brief Gets the number of poles in a DLR.
 *
 * This function returns the number of poles in the specified DLR object.
 *
 * @param dlr Pointer to the DLR object
 * @param num_poles Pointer to store the number of poles
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @see spir_dlr_get_poles
 */
int32_t spir_dlr_get_num_poles(const spir_dlr *dlr, int32_t *num_poles);

/**
 * @brief Gets the poles in a DLR.
 *
 * This function returns the poles in the specified DLR object.
 *
 * @param dlr Pointer to the DLR object
 * @param poles Pointer to store the poles
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @see spir_dlr_get_num_poles
 */
int32_t spir_dlr_get_poles(const spir_dlr *dlr, double *poles);

/**
 * @brief Transforms a given input array from the Intermediate Representation (IR)
 * to the Discrete Lehmann Representation (DLR) using the specified DLR object.
 * This version handles real (double precision) input and output arrays.
 *
 * @param dlr Pointer to the fermionic DLR object
 * @param order Order type (C or Fortran)
 * @param ndim Number of dimensions of input/output arrays
 * @param input_dims Array of dimensions
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input coefficients array in IR
 * @param out Output array in DLR
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The input and output arrays must be allocated with sufficient memory.
 *       The size of the input and output arrays should match the dimensions
 *       specified. The order type determines the memory layout of the input and
 *       output arrays. The function assumes that the input array is in the
 *       specified order type. The output array will be in the specified order
 *       type.
 *
 * @see spir_dlr_from_IR
 * @see spir_dlr_to_IR_dd
 */
int32_t spir_dlr_from_IR_dd(const spir_dlr *dlr, spir_order_type order,
                         int32_t ndim, int32_t *input_dims, int32_t target_dim,
                         const double *input, double *out);

int32_t spir_dlr_from_IR_zz(const spir_dlr *dlr, spir_order_type order,
                         int32_t ndim, int32_t *input_dims, int32_t target_dim,
                         const c_complex *input, c_complex *out);


/**
 * @brief Transforms coefficients from DLR basis to IR representation.
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
 * @param dlr Pointer to the DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of DLR coefficients (double precision)
 * @param out Output array for the IR coefficients (double precision)
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note The transformation is a direct matrix multiplication, which is
 *       typically faster than the inverse transformation
 *
 * @see spir_dlr_from_IR
 */
int32_t spir_dlr_to_IR_dd(const spir_dlr *dlr, spir_order_type order, int32_t ndim,
                       int32_t *input_dims, int32_t target_dim,
                       const double *input, double *out);

/**
 * @brief Transforms coefficients from DLR basis to IR representation.
 * This version handles complex input and output arrays.
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
 * @param dlr Pointer to the DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of DLR coefficients (complex)
 * @param out Output array for the IR coefficients (complex)
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note The transformation is a direct matrix multiplication, which is
 *       typically faster than the inverse transformation
 *
 * @see spir_dlr_from_IR_zz
 * @see spir_dlr_to_IR_dd
 */
int32_t spir_dlr_to_IR_zz(const spir_dlr *dlr, spir_order_type order, int32_t ndim,
                       int32_t *input_dims, int32_t target_dim,
                       const c_complex *input, c_complex *out);

/**
 * @brief Gets the basis functions of a DLR.
 *
 * This function returns an object representing the basis functions
 * in the imaginary-time domain of the specified DLR object.
 *
 * @param dlr Pointer to the DLR object
 * @param u Pointer to store the basis functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 * @see spir_destroy_funcs
 */
int32_t spir_dlr_get_u(const spir_dlr *dlr, spir_funcs **u);

/**
 * @brief Gets the basis functions of a DLR in the Matsubara-frequency domain.
 *
 * This function returns an object representing the basis functions
 * in the Matsubara-frequency domain of the specified DLR object.
 *
 * @param dlr Pointer to the DLR object
 * @param uhat Pointer to store the basis functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 */
int32_t spir_dlr_get_uhat(const spir_dlr *dlr, spir_matsubara_funcs **uhat);

/**
 * @brief Creates a new finite temperature IR basis.
 *
 * This function creates a new finite temperature IR basis using the specified
 * parameters and the default logistic kernel. The basis is constructed based on
 * the given beta (inverse temperature), omega_max (frequency cutoff), and
 * epsilon (accuracy target).
 *
 * @param b Pointer to a pointer that will store the newly created basis object.
 *          On successful creation, this will point to the new basis object.
 *          If creation fails, this will be set to NULL.
 * @param statistics Statistics type (SPIR_STATISTICS_FERMIONIC or
 * SPIR_STATISTICS_BOSONIC)
 * @param beta Inverse temperature β (must be positive)
 * @param omega_max Frequency cutoff ωmax (must be non-negative)
 * @param epsilon Accuracy target for the basis
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 * @see spir_finite_temp_basis_new_with_kernel
 */
int32_t spir_finite_temp_basis_new(spir_finite_temp_basis **b, spir_statistics_type statistics, double beta,
                           double omega_max, double epsilon);

/**
 * @brief Creates a new finite temperature IR basis using a
 * pre-computed SVE result.
 *
 * This function creates a intermediate representation (IR) basis
 * using a pre-computed singular value expansion (SVE) result. This allows for
 * reusing an existing SVE computation, which can be more efficient than
 * recomputing it.
 *
 * @param b Pointer to a pointer that will store the newly created basis object.
 *          On successful creation, this will point to the new basis object.
 *          If creation fails, this will be set to NULL.
 * @param statistics Statistics type (SPIR_STATISTICS_FERMIONIC or
 * SPIR_STATISTICS_BOSONIC)
 * @param beta Inverse temperature β (must be positive)
 * @param omega_max Frequency cutoff ωmax (must be non-negative)
 * @param k Pointer to the kernel object used for the basis construction
 * @param sve Pointer to a pre-computed SVE result for the kernel
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note Using a pre-computed SVE can significantly improve performance when
 *       creating multiple basis objects with the same kernel
 * @see spir_sve_result_new
 * @see spir_destroy_finite_temp_basis
 */
int32_t spir_finite_temp_basis_new_with_sve(
    spir_finite_temp_basis **b, spir_statistics_type statistics, double beta, double omega_max,
    const spir_kernel *k, const spir_sve_result *sve);

/**
 * @brief Gets the basis functions of a finite temperature basis.
 *
 * This function returns an object representing the basis functions
 * in the imaginary-time domain of the specified finite temperature basis.
 *
 * @param b Pointer to the finite temperature basis object
 * @param u Pointer to store the basis functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The returned object must be freed using spir_destroy_funcs
 *       when no longer needed
 * @see spir_destroy_funcs
 */
int32_t spir_finite_temp_basis_get_u(const spir_finite_temp_basis *b,
                                     spir_funcs **u);

/**
 * @brief Gets the basis functions of a finite temperature basis.
 *
 * This function returns an object representing the basis functions
 * in the real-frequency domain of the specified finite temperature basis.
 *
 * @param b Pointer to the finite temperature basis object
 * @param v Pointer to store the basis functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The returned object must be freed using spir_destroy_funcs
 *       when no longer needed
 * @see spir_destroy_funcs
 */
int32_t spir_finite_temp_basis_get_v(const spir_finite_temp_basis *b,
                                     spir_funcs **v);

/**
 * @brief Gets the basis functions of a finite temperature basis in Matsubara
 * frequency domain.
 *
 * This function returns an object representing the basis functions
 * in the Matsubara-frequency domain of the specified finite temperature basis.
 *
 * @param b Pointer to the finite temperature basis object
 * @param uhat Pointer to store the basis functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The returned object must be freed using spir_destroy_matsubara_funcs
 *       when no longer needed
 * @see spir_destroy_matsubara_funcs
 */
int32_t spir_finite_temp_basis_get_uhat(const spir_finite_temp_basis *b,
                                        spir_matsubara_funcs **uhat);

/**
 * @brief Gets the number of functions in a functions object.
 *
 * This function returns the number of functions contained in the specified
 * functions object. This number is needed to allocate arrays of the correct
 * size when evaluating the functions.
 *
 * @param funcs Pointer to the functions object
 * @param size Pointer to store the number of functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 */
int32_t spir_funcs_get_size(const spir_funcs *funcs, int32_t *size);

/**
 * @brief Evaluates functions at a single point in the imaginary-time domain or
 * the real frequency domain.
 *
 * This function evaluates all functions at a specified point x.
 * The values of each basis function at x are stored in the output array.
 * The output array out[j] contains the value of the j-th function evaluated at
 * x.
 *
 * @param uv Pointer to a functions object
 * @param x Point at which to evaluate the functions
 * @param out Pre-allocated array to store the evaluation results.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with sufficient size to store
 *       all function values
 */
int32_t spir_evaluate_funcs(const spir_funcs *funcs, double x, double *out);

/**
 * @brief Evaluates basis functions at multiple Matsubara frequencies.
 *
 * This function evaluates all functions contained in a Matsubara functions
 * object at the specified Matsubara frequency indices. The values of each
 * function at each frequency are stored in the output array.
 *
 * @param uiw Pointer to the Matsubara functions object
 * @param order Specifies the memory layout of the output array:
 *             SPIR_ORDER_ROW_MAJOR for row-major order (frequency index varies
 * fastest), SPIR_ORDER_COLUMN_MAJOR for column-major order (function index
 * varies fastest)
 * @param num_freqs Number of Matsubara frequencies at which to evaluate
 * @param matsubara_freq_indices Array of Matsubara frequency indices
 * @param out Pre-allocated array to store the evaluation results. The results
 * are stored as a 2D array of size num_freqs x n_funcs.
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The output array must be pre-allocated with sufficient size to store
 *       all function values at all requested frequencies. Indices n correspond to
 *       ωn = nπ/β, where n are odd for fermionic frequencies and even for
 *       bosonic frequencies.
 */
int32_t spir_evaluate_matsubara_funcs(const spir_matsubara_funcs *uiw,
                                      spir_order_type order, int32_t num_freqs,
                                      int32_t *matsubara_freq_indices,
                                      c_complex *out);

/**
 * @brief Gets the number of sampling points in a sampling object.
 *
 * This function returns the number of sampling points used in the specified
 * sampling object. This number is needed to allocate arrays of the correct size
 * when retrieving the actual sampling points.
 *
 * @param s Pointer to the sampling object
 * @param num_points Pointer to store the number of sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @see spir_sampling_get_tau_points
 * @see spir_sampling_get_matsubara_points
 */
int32_t spir_sampling_get_num_points(const spir_sampling *s,
                                     int32_t *num_points);

/**
 * @brief Gets the imaginary time sampling points.
 *
 * This function fills the provided array with the imaginary time (τ) sampling
 * points used in the specified sampling object. The array must be pre-allocated
 * with sufficient size (use spir_sampling_get_num_points to determine the
 * required size).
 *
 * @param s Pointer to the sampling object
 * @param points Pre-allocated array to store the τ sampling points
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The array must be pre-allocated with size >=
 *       spir_sampling_get_num_points(s)
 * @see spir_sampling_get_num_points
 */
int32_t spir_sampling_get_tau_points(const spir_sampling *s, double *points);

/**
 * @brief Gets the Matsubara frequency sampling points.
 *
 * This function fills the provided array with the Matsubara frequency indices
 * (n) used in the specified sampling object. The actual Matsubara frequencies
 * are ωn = (2n + 1)π/β for fermionic case and ωn = 2nπ/β for bosonic case. The
 * array must be pre-allocated with sufficient size (use
 * spir_sampling_get_num_points to determine the required size).
 *
 * @param s Pointer to the sampling object
 * @param points Pre-allocated array to store the Matsubara frequency indices
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 *
 * @note The array must be pre-allocated with size >=
 *       spir_sampling_get_num_points(s)
 * @note For fermionic case, the indices n give frequencies ωn = (2n + 1)π/β
 * @note For bosonic case, the indices n give frequencies ωn = 2nπ/β
 * @see spir_sampling_get_num_points
 */
int32_t spir_sampling_get_matsubara_points(const spir_sampling *s,
                                           int32_t *points);

/**
 * @brief Gets the number of functions in a Matsubara functions object.
 *
 * This function returns the number of functions contained in the specified
 * Matsubara functions object. This number is needed to allocate arrays of the
 * correct size when evaluating the functions.
 *
 * @param funcs Pointer to the Matsubara functions object
 * @param size Pointer to store the number of functions
 * @return An integer status code:
 *         - 0 (SPIR_COMPUTATION_SUCCESS) on success
 *         - A non-zero error code on failure
 */
int32_t spir_matsubara_funcs_get_size(const spir_matsubara_funcs *funcs,
                                      int32_t *size);

#ifdef __cplusplus
}
#endif
