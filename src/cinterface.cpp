#include "sparseir/sparseir.h"
#include "sparseir/sparseir.hpp"
#include "sparseir/utils.hpp"
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <iostream>

// Debug macro
#ifdef DEBUG_CINTERFACE
#define DEBUG_LOG(msg) std::cerr << "[DEBUG] " << msg << std::endl
#else
#define DEBUG_LOG(msg)
#endif

// Define opaque type and implement its management functions
#define IMPLEMENT_OPAQUE_TYPE(name, impl_type)                                 \
    struct _spir_##name                                                        \
    {                                                                          \
        std::shared_ptr<impl_type> ptr;                                        \
        _spir_##name() { }                                                     \
        ~_spir_##name() { }                                                    \
    };                                                                         \
    typedef struct _spir_##name spir_##name;                                   \
                                                                               \
    /* Helper function for creating objects */                                 \
    static inline spir_##name *create_##name(                                  \
        std::shared_ptr<impl_type> p)                                          \
    {                                                                          \
        auto *obj = new spir_##name;                                           \
        obj->ptr = p;                                                          \
        return obj;                                                            \
    }                                                                          \
                                                                               \
    /* Check if the shared_ptr has a valid object */                           \
    int spir_is_assigned_##name(const spir_##name *obj)                        \
    {                                                                          \
        if (!obj) {                                                            \
            DEBUG_LOG(#name << " object is null");                             \
            return 0;                                                          \
        }                                                                      \
        bool is_assigned = static_cast<bool>(obj->ptr);                        \
        DEBUG_LOG(#name << " object at " << obj << ", ptr=" << obj->ptr.get() << ", is_assigned=" << is_assigned); \
        return is_assigned ? 1 : 0;                                            \
    }                                                                          \
                                                                               \
    /* Clone function */                                                       \
    spir_##name *spir_clone_##name(const spir_##name *src)                   \
    {                                                                          \
        DEBUG_LOG("Cloning " << #name << " at " << src);                       \
        if (!src) {                                                            \
            DEBUG_LOG("Source " << #name << " is null");                       \
            return nullptr;                                                    \
        }                                                                      \
                                                                               \
        try {                                                                  \
            /* Create a new structure */                                       \
            spir_##name *result = new spir_##name();                           \
                                                                               \
            /* If source has a valid shared_ptr, copy it */                    \
            if (src->ptr) {                                                    \
                /* Create a new shared_ptr instance that shares ownership */   \
                result->ptr = src->ptr;                                        \
                DEBUG_LOG("Cloned " << #name << " to " << result << ", shared_ptr points to " << result->ptr.get()); \
            } else {                                                           \
                DEBUG_LOG("Source " << #name << " has null shared_ptr");       \
                result->ptr = nullptr;                                         \
            }                                                                  \
                                                                               \
            return result;                                                     \
        } catch (const std::exception& e) {                                    \
            DEBUG_LOG("Exception in " << #name << "_clone: " << e.what());     \
            return nullptr;                                                    \
        } catch (...) {                                                        \
            DEBUG_LOG("Unknown exception in " << #name << "_clone");           \
            return nullptr;                                                    \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Destroy function */                                                     \
    void spir_destroy_##name(spir_##name *obj)                                 \
    {                                                                          \
        if (!obj) {                                                            \
            DEBUG_LOG(#name << " object is null");                             \
            return;                                                            \
        }                                                                      \
        DEBUG_LOG("Destroying " << #name << " object at " << obj);             \
        /* Check before resetting */                                           \
        if (obj->ptr) {                                                        \
            DEBUG_LOG("Resetting shared_ptr in " << #name << " at " << obj->ptr.get()); \
            obj->ptr.reset();                                                  \
        }                                                                      \
        /* Safely delete the object */                                         \
        delete obj;                                                            \
    }                                                                          \
                                                                               \
    /* Helper to get the implementation shared_ptr */                          \
    static inline std::shared_ptr<impl_type> get_impl_##name(const spir_##name *obj) \
    {                                                                          \
        if (!obj) {                                                            \
            DEBUG_LOG(#name << " object is null");                             \
            return nullptr;                                                    \
        }                                                                      \
        DEBUG_LOG(#name << " object at " << obj << ", ptr=" << obj->ptr.get());\
        return obj->ptr;                                                       \
    }

// Implementation of the opaque types
IMPLEMENT_OPAQUE_TYPE(kernel, sparseir::AbstractKernel);
IMPLEMENT_OPAQUE_TYPE(logistic_kernel, sparseir::LogisticKernel);
IMPLEMENT_OPAQUE_TYPE(regularized_bose_kernel, sparseir::RegularizedBoseKernel);
IMPLEMENT_OPAQUE_TYPE(polyvector, sparseir::PiecewiseLegendrePolyVector);

IMPLEMENT_OPAQUE_TYPE(basis, sparseir::FiniteTempBasis<sparseir::Fermionic>);
IMPLEMENT_OPAQUE_TYPE(fermionic_finite_temp_basis,
                      sparseir::FiniteTempBasis<sparseir::Fermionic>);
IMPLEMENT_OPAQUE_TYPE(bosonic_finite_temp_basis,
                      sparseir::FiniteTempBasis<sparseir::Bosonic>);
IMPLEMENT_OPAQUE_TYPE(sampling, sparseir::AbstractSampling);
IMPLEMENT_OPAQUE_TYPE(sve_result, sparseir::SVEResult);
IMPLEMENT_OPAQUE_TYPE(fermionic_dlr, sparseir::DiscreteLehmannRepresentation<sparseir::Fermionic>);
IMPLEMENT_OPAQUE_TYPE(bosonic_dlr, sparseir::DiscreteLehmannRepresentation<sparseir::Bosonic>);

// Helper function to convert N-dimensional array to 3D array by collapsing dimensions
static std::array<int32_t, 3> collapse_to_3d(int32_t ndim, const int32_t* dims, int32_t target_dim) {
    std::array<int32_t, 3> dims_3d = {1, dims[target_dim], 1};
    // Multiply all dimensions before target_dim into first dimension
    for (int32_t i = 0; i < target_dim; ++i) {
        dims_3d[0] *= dims[i];
    }
    // Multiply all dimensions after target_dim into last dimension
    for (int32_t i = target_dim + 1; i < ndim; ++i) {
        dims_3d[2] *= dims[i];
    }
    return dims_3d;
}

// Helper function to convert N-dimensional array to 2D array by collapsing dimensions
static std::array<int32_t, 2> collapse_to_2d(int32_t ndim, const int32_t* dims, int32_t target_dim) {
    std::array<int32_t, 2> dims_2d = {dims[target_dim], 1};
    // Multiply all dimensions before target_dim into first dimension
    for (int32_t i = 0; i < target_dim; ++i) {
        dims_2d[0] *= dims[i];
    }
    // Multiply all dimensions after target_dim into last dimension
    for (int32_t i = target_dim + 1; i < ndim; ++i) {
        dims_2d[1] *= dims[i];
    }
    return dims_2d;
}

// Template function to handle all evaluation cases - moved outside extern "C" block
template<typename InputScalar, typename OutputScalar>
static int evaluate_impl(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const InputScalar *input,
    OutputScalar *out,
    int (sparseir::AbstractSampling::*eval_func)(
        const Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> &,
        int,
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> &) const)
{
    auto impl = get_impl_sampling(s);
    if (!impl)
        return -1;

    // Convert dimensions
    std::array<int32_t, 3> dims_3d =
        collapse_to_3d(ndim, input_dims, target_dim);

    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());
        std::array<int32_t, 3> output_dims_3d = input_dims_3d;

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(out, output_dims_3d);
        // Convert to column-major order for Eigen
        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    } else{
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::array<int32_t, 3> output_dims_3d = input_dims_3d;
        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(input,
                                                                       input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(out,
                                                                   output_dims_3d);

        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    }
}

template<typename InputScalar, typename OutputScalar>
static int fit_impl(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const InputScalar *input,
    OutputScalar *out,
    int (sparseir::AbstractSampling::*eval_func)(
        const Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> &,
        int,
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> &) const)
{
    auto impl = get_impl_sampling(s);
    if (!impl)
        return -1;

    // Convert dimensions
    std::array<int32_t, 3> dims_3d =
        collapse_to_3d(ndim, input_dims, target_dim);

    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());

        std::array<int32_t, 3> output_dims_3d = input_dims_3d;

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(out, output_dims_3d);
        // Convert to column-major order for Eigen
        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    } else{
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::array<int32_t, 3> output_dims_3d = input_dims_3d;

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(input,
                                                                       input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(out,
                                                                   output_dims_3d);

        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    }
}

// Implementation of the C API
extern "C" {

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
spir_kernel *spir_logistic_kernel_new(double lambda)
{
    DEBUG_LOG("Creating LogisticKernel with lambda=" << lambda);
    try {
        auto kernel = std::make_shared<sparseir::LogisticKernel>(lambda);
        auto abstract_kernel = std::shared_ptr<sparseir::AbstractKernel>(kernel);
        auto result = create_kernel(abstract_kernel);
        DEBUG_LOG("Created LogisticKernel at " << result << ", ptr=" << result->ptr.get());
        return result;
    } catch (const std::exception& e) {
        DEBUG_LOG("Exception in spir_logistic_kernel_new: " << e.what());
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_logistic_kernel_new");
        return nullptr;
    }
}

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
spir_kernel *spir_regularized_bose_kernel_new(double lambda)
{
    DEBUG_LOG("Creating RegularizedBoseKernel with lambda=" << lambda);
    try {
        auto kernel = std::make_shared<sparseir::RegularizedBoseKernel>(lambda);
        auto abstract_kernel = std::shared_ptr<sparseir::AbstractKernel>(kernel);
        auto result = create_kernel(abstract_kernel);
        DEBUG_LOG("Created RegularizedBoseKernel at " << result << ", ptr=" << result->ptr.get());
        return result;
    } catch (const std::exception& e) {
        DEBUG_LOG("Exception in spir_regularized_bose_kernel_new: " << e.what());
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_regularized_bose_kernel_new");
        return nullptr;
    }
}

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
 * @return 0 on success, -1 on failure (if the kernel is invalid or an exception occurs)
 */
int spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                       double *ymin, double *ymax)
{
    DEBUG_LOG("spir_kernel_domain called with kernel=" << k);
    auto impl = get_impl_kernel(k);
    if (!impl) {
        DEBUG_LOG("Failed to get kernel implementation");
        return -1;
    }

    try {
        DEBUG_LOG("Getting xrange and yrange");
        auto xrange = impl->xrange();
        auto yrange = impl->yrange();

        DEBUG_LOG("Setting output values: xrange=(" << xrange.first << ", " << xrange.second << "), yrange=(" << yrange.first << ", " << yrange.second << ")");
        *xmin = xrange.first;
        *xmax = xrange.second;
        *ymin = yrange.first;
        *ymax = yrange.second;

        return 0;
    } catch (const std::exception& e) {
        DEBUG_LOG("Exception in spir_kernel_domain: " << e.what());
        return -1;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_kernel_domain");
        return -1;
    }
}

/*
int spir_kernel_evaluate(const spir_kernel *k, double x, double y, double *out)
{
    auto impl = get_impl_kernel(k);
    if (!impl || !out)
        return -1;

    try {
        *out = impl->compute(x, y);
        return 0;
    } catch (...) {
        return -1;
    }
}

int spir_kernel_matrix(const spir_kernel *k, const double *x, int nx,
                       const double *y, int ny, double *out)
{
    auto impl = get_impl_kernel(k);
    if (!impl || !x || !y || !out)
        return -1;
    if (nx <= 0 || ny <= 0)
        return -1;

    try {
        // Evaluate kernel at each point
        for (int ix = 0; ix < nx; ++ix) {
            for (int iy = 0; iy < ny; ++iy) {
                out[ix * ny + iy] =
                    impl->compute(x[ix], y[iy]); // column-major order
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}
*/

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
 * - u[l](x) are left singular functions, forming an orthonormal system on [xmin, xmax]
 * - v[l](y) are right singular functions, forming an orthonormal system on [ymin, ymax]
 *
 * The SVE is computed by mapping it onto a singular value decomposition (SVD)
 * of a matrix using piecewise Legendre polynomial expansion.
 *
 * @param k Pointer to the kernel object for which to compute SVE
 * @param epsilon Accuracy target for the basis. Determines:
 *               - The relative magnitude of included singular values
 *               - The accuracy of computed singular values and vectors
 *
 * @return A pointer to the newly created SVE result object containing the truncated
 *         singular value expansion, or NULL if creation fails
 *
 * @note The computation automatically uses optimized strategies:
 *       - For centrosymmetric kernels, specialized algorithms are employed
 *       - The working precision is adjusted to meet accuracy requirements
 *
 * @note The returned object must be freed using spir_destroy_sve_result when no longer needed
 * @see spir_destroy_sve_result
 */
spir_sve_result* spir_sve_result_new(const spir_kernel* k, double epsilon)
{
    try {
        auto impl = get_impl_kernel(k);
        if (!impl)
            return nullptr;
        auto sve = sparseir::compute_sve(impl, epsilon);
        return create_sve_result(std::make_shared<sparseir::SVEResult>(sve));
    } catch (...) {
        return nullptr;
    }
}

/**
 * @brief Creates a new fermionic finite temperature IR basis.
 *
 * For a continuation kernel K from real frequencies, ω ∈ [-ωmax, ωmax], to
 * imaginary time, τ ∈ [0, β], this function creates an intermediate representation (IR)
 * basis that stores the truncated singular value expansion:
 *
 * K(τ, ω) ≈ ∑ u[l](τ) * s[l] * v[l](ω) for l = 1, 2, 3, ...
 *
 * where:
 * - u[l](τ) are IR basis functions on the imaginary time axis (stored as piecewise Legendre polynomials)
 * - s[l] are singular values of the continuation kernel
 * - v[l](ω) are IR basis functions on the real frequency axis (stored as piecewise Legendre polynomials)
 *
 * @param beta Inverse temperature β (must be positive)
 * @param omega_max Frequency cutoff ωmax (must be non-negative)
 * @param epsilon Accuracy target for the basis
 *
 * @return A pointer to the newly created fermionic finite temperature basis object,
 *         or NULL if creation fails
 *
 * @note The basis includes both imaginary time and Matsubara frequency representations
 * @note The returned object must be freed using spir_destroy_fermionic_finite_temp_basis when no longer needed
 * @see spir_destroy_fermionic_finite_temp_basis
 */
spir_fermionic_finite_temp_basis *
spir_fermionic_finite_temp_basis_new(double beta, double omega_max,
                                     double epsilon)
{
    try {
        return create_fermionic_finite_temp_basis(
            std::make_shared<sparseir::FiniteTempBasis<sparseir::Fermionic>>(
                beta, omega_max, epsilon));
    } catch (...) {
        return nullptr;
    }
}

/**
 * @brief Creates a new bosonic finite temperature IR basis.
 *
 * For a continuation kernel K from real frequencies, ω ∈ [-ωmax, ωmax], to
 * imaginary time, τ ∈ [0, β], this function creates an intermediate representation (IR)
 * basis that stores the truncated singular value expansion:
 *
 * K(τ, ω) ≈ ∑ u[l](τ) * s[l] * v[l](ω) for l = 1, 2, 3, ...
 *
 * where:
 * - u[l](τ) are IR basis functions on the imaginary time axis (stored as piecewise Legendre polynomials)
 * - s[l] are singular values of the continuation kernel
 * - v[l](ω) are IR basis functions on the real frequency axis (stored as piecewise Legendre polynomials)
 *
 * @param beta Inverse temperature β (must be positive)
 * @param omega_max Frequency cutoff ωmax (must be non-negative)
 * @param epsilon Accuracy target for the basis
 *
 * @return A pointer to the newly created bosonic finite temperature basis object,
 *         or NULL if creation fails
 *
 * @note The basis includes both imaginary time and Matsubara frequency representations
 * @note For Matsubara frequencies, bosonic basis uses even numbers (2n)
 * @note The returned object must be freed using spir_destroy_bosonic_finite_temp_basis when no longer needed
 * @see spir_destroy_bosonic_finite_temp_basis
 */
spir_bosonic_finite_temp_basis *
spir_bosonic_finite_temp_basis_new(double beta, double omega_max,
                                     double epsilon)
{
    try {
        return create_bosonic_finite_temp_basis(
            std::make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
                beta, omega_max, epsilon));
    } catch (...) {
        return nullptr;
    }
}

/**
 * @brief Creates a new fermionic finite temperature IR basis using a pre-computed SVE result.
 *
 * This function creates a fermionic intermediate representation (IR) basis using
 * a pre-computed singular value expansion (SVE) result. This allows for reusing
 * an existing SVE computation, which can be more efficient than recomputing it.
 *
 * @param beta Inverse temperature β (must be positive)
 * @param omega_max Frequency cutoff ωmax (must be non-negative)
 * @param k Pointer to the kernel object used for the basis construction
 * @param sve Pointer to a pre-computed SVE result for the kernel
 *
 * @return A pointer to the newly created fermionic finite temperature basis object,
 *         or NULL if creation fails (invalid inputs or exception occurs)
 *
 * @note Using a pre-computed SVE can significantly improve performance when creating
 *       multiple basis objects with the same kernel
 * @see spir_sve_result_new
 * @see spir_destroy_fermionic_finite_temp_basis
 */
spir_fermionic_finite_temp_basis *
spir_fermionic_finite_temp_basis_new_with_sve(double beta, double omega_max,
                                             const spir_kernel *k,
                                             const spir_sve_result *sve)
{
    try {
        auto sve_impl = get_impl_sve_result(sve);
        auto kernel_impl = get_impl_kernel(k);
        if (!sve_impl || !kernel_impl)
            return nullptr;
        return create_fermionic_finite_temp_basis(
            std::make_shared<sparseir::FiniteTempBasis<sparseir::Fermionic>>(
                beta, omega_max, kernel_impl, *sve_impl));
    } catch (...) {
        return nullptr;
    }
}

/**
 * @brief Creates a new bosonic finite temperature IR basis using a pre-computed SVE result.
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
 * @return A pointer to the newly created bosonic finite temperature basis object,
 *         or NULL if creation fails (invalid inputs or exception occurs)
 *
 * @note Using a pre-computed SVE can significantly improve performance when creating
 *       multiple basis objects with the same kernel
 * @see spir_sve_result_new
 * @see spir_destroy_bosonic_finite_temp_basis
 */
spir_bosonic_finite_temp_basis *
spir_bosonic_finite_temp_basis_new_with_sve(double beta, double omega_max,
                                              const spir_kernel *k,
                                              const spir_sve_result *sve)
{
    try {
        auto sve_impl = get_impl_sve_result(sve);
        auto kernel_impl = get_impl_kernel(k);
        if (!sve_impl || !kernel_impl)
            return nullptr;
        return create_bosonic_finite_temp_basis(
            std::make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
                beta, omega_max, kernel_impl, *sve_impl));
    } catch (...) {
        return nullptr;
    }
}

/**
 * @brief Creates a new fermionic tau sampling object for sparse sampling in imaginary time.
 *
 * Constructs a sampling object that allows transformation between the IR basis and
 * a set of sampling points in imaginary time (τ). The sampling points are automatically
 * chosen as the extrema of the highest-order basis function in imaginary time, which
 * provides near-optimal conditioning for the given basis size.
 *
 * @param b Pointer to a fermionic finite temperature basis object
 * @return A pointer to the newly created sampling object, or NULL if creation fails
 *
 * @note The sampling points are chosen to optimize numerical stability and accuracy
 * @note The sampling matrix is automatically factorized using SVD for efficient transformations
 * @note The returned object must be freed using spir_destroy_sampling when no longer needed
 * @see spir_destroy_sampling
 */
spir_sampling *
spir_fermionic_tau_sampling_new(const spir_fermionic_finite_temp_basis *b)
{
    auto impl = get_impl_fermionic_finite_temp_basis(b);
    if (!impl)
        return nullptr;
    auto smpl = std::make_shared<sparseir::TauSampling<sparseir::Fermionic>>(*impl);
    return create_sampling(smpl);
}

/**
 * @brief Creates a new fermionic Matsubara sampling object for sparse sampling in Matsubara frequencies.
 *
 * Constructs a sampling object that allows transformation between the IR basis and
 * a set of sampling points in Matsubara frequencies (iωn). The sampling points are
 * automatically chosen as the (discrete) extrema of the highest-order basis function
 * in Matsubara frequencies, which provides near-optimal conditioning for the given basis size.
 *
 * For fermionic Matsubara frequencies, the sampling points are odd integers:
 * iωn = (2n + 1)π/β, where n is an integer.
 *
 * @param b Pointer to a fermionic finite temperature basis object
 * @return A pointer to the newly created sampling object, or NULL if creation fails
 *
 * @note The sampling points are chosen to optimize numerical stability and accuracy
 * @note The sampling matrix is automatically factorized using SVD for efficient transformations
 * @note For fermionic functions, the Matsubara frequencies are odd multiples of π/β
 * @note The returned object must be freed using spir_destroy_sampling when no longer needed
 * @see spir_destroy_sampling
 */
spir_sampling *
spir_fermionic_matsubara_sampling_new(const spir_fermionic_finite_temp_basis *b)
{
    auto impl = get_impl_fermionic_finite_temp_basis(b);
    if (!impl)
        return nullptr;
    auto smpl = std::make_shared<sparseir::MatsubaraSampling<sparseir::Fermionic>>(*impl);
    return create_sampling(smpl);
}

/**
 * @brief Creates a new fermionic Discrete Lehmann Representation (DLR).
 *
 * This function implements a variant of the discrete Lehmann representation (DLR).
 * Unlike the IR which uses truncated singular value expansion of the analytic
 * continuation kernel K, the DLR is based on a "sketching" of K. The resulting basis
 * is a linear combination of discrete set of poles on the real-frequency axis,
 * continued to the imaginary-frequency axis:
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
 * @note The returned object must be freed using spir_destroy_fermionic_dlr when no longer needed
 * @see spir_destroy_fermionic_dlr
 * @see spir_fermionic_dlr_new_with_poles
 *
 * @warning This implementation uses a heuristic approach for pole selection, which
 *          differs from the original DLR method that uses rank-revealing decomposition
 */
spir_fermionic_dlr *
spir_fermionic_dlr_new(const spir_fermionic_finite_temp_basis *b)
{
    auto impl = get_impl_fermionic_finite_temp_basis(b);
    if (!impl)
        return nullptr;
    auto dlr = std::make_shared<sparseir::DiscreteLehmannRepresentation<sparseir::Fermionic>>(*impl);
    return create_fermionic_dlr(dlr);
}

/**
 * @brief Creates a new fermionic Discrete Lehmann Representation (DLR) with custom poles.
 *
 * This function creates a fermionic DLR using a set of user-specified poles on the
 * real-frequency axis. The DLR represents Green's functions as a sum of poles:
 *
 * G(iν) = ∑ a[i] / (iν - w[i]) for i = 1, 2, ..., npoles
 *
 * where w[i] are the specified poles and a[i] are the expansion coefficients.
 *
 * @param b Pointer to a fermionic finite temperature basis object
 * @param npoles Number of poles to use in the representation
 * @param poles Array of pole locations on the real-frequency axis
 *
 * @return A pointer to the newly created DLR object with custom poles, or NULL if creation fails
 *
 * @note This function allows for more control over the pole selection compared to the
 *       automatic pole selection in spir_fermionic_dlr_new
 * @see spir_fermionic_dlr_new
 * @see spir_destroy_fermionic_dlr
 */
spir_fermionic_dlr *
spir_fermionic_dlr_new_with_poles(
    const spir_fermionic_finite_temp_basis *b, const int npoles, const double *poles
) {
    auto impl = get_impl_fermionic_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    Eigen::VectorXd poles_vec(npoles);
    for (int i = 0; i < npoles; i++) {
        poles_vec(i) = poles[i];
    }
    auto dlr = std::make_shared<sparseir::DiscreteLehmannRepresentation<sparseir::Fermionic>>(*impl, poles_vec);
    return create_fermionic_dlr(dlr);
}

/**
 * @brief Creates a new bosonic Discrete Lehmann Representation (DLR).
 *
 * This function implements a variant of the discrete Lehmann representation (DLR).
 * Unlike the IR which uses truncated singular value expansion of the analytic
 * continuation kernel K, the DLR is based on a "sketching" of K. The resulting basis
 * is a linear combination of discrete set of poles on the real-frequency axis,
 * continued to the imaginary-frequency axis:
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
 * @note The returned object must be freed using spir_destroy_bosonic_dlr when no longer needed
 * @see spir_destroy_bosonic_dlr
 * @see spir_bosonic_dlr_new_with_poles
 *
 * @warning This implementation uses a heuristic approach for pole selection, which
 *          differs from the original DLR method that uses rank-revealing decomposition
 */
spir_bosonic_dlr *
spir_bosonic_dlr_new(const spir_bosonic_finite_temp_basis *b)
{
    auto impl = get_impl_bosonic_finite_temp_basis(b);
    if (!impl)
        return nullptr;
    auto dlr = std::make_shared<sparseir::DiscreteLehmannRepresentation<sparseir::Bosonic>>(*impl);
    return create_bosonic_dlr(dlr);
}

/**
 * @brief Creates a new bosonic Discrete Lehmann Representation (DLR) with custom poles.
 *
 * This function creates a bosonic DLR using a set of user-specified poles on the
 * real-frequency axis. The DLR represents correlation functions as a sum of poles:
 *
 * G(iωn) = ∑ a[i] / (iωn - w[i]) for i = 1, 2, ..., npoles
 *
 * where w[i] are the specified poles and a[i] are the expansion coefficients.
 *
 * @param b Pointer to a bosonic finite temperature basis object
 * @param npoles Number of poles to use in the representation
 * @param poles Array of pole locations on the real-frequency axis
 *
 * @return A pointer to the newly created DLR object with custom poles, or NULL if creation fails
 *
 * @note This function allows for more control over the pole selection compared to the
 *       automatic pole selection in spir_bosonic_dlr_new
 * @see spir_bosonic_dlr_new
 * @see spir_destroy_bosonic_dlr
 */
spir_bosonic_dlr *
spir_bosonic_dlr_new_with_poles(
    const spir_bosonic_finite_temp_basis *b, const int npoles, const double *poles
) {
    auto impl = get_impl_bosonic_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    Eigen::VectorXd poles_vec(npoles);
    for (int i = 0; i < npoles; i++) {
        poles_vec(i) = poles[i];
    }
    auto dlr = std::make_shared<sparseir::DiscreteLehmannRepresentation<sparseir::Bosonic>>(*impl, poles_vec);
    return create_bosonic_dlr(dlr);
}

/**
 * @brief Evaluates basis coefficients at sampling points (double to double version).
 *
 * Transforms basis coefficients to values at sampling points, where both input and
 * output are real (double precision) values. The operation can be performed along
 * any dimension of a multidimensional array.
 *
 * @param s Pointer to the sampling object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of basis coefficients
 * @param out Output array for the evaluated values at sampling points
 *
 * @return 0 on success, non-zero on failure
 *
 * @note For optimal performance, the target dimension should be either the first (0)
 *       or the last (ndim-1) dimension to avoid large temporary array allocations
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 *
 * @see spir_sampling_evaluate_dz
 * @see spir_sampling_evaluate_zz
 */
int spir_sampling_evaluate_dd(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const double *input,
    double *out)
{
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, out,
                        &sparseir::AbstractSampling::evaluate_inplace_dd);
}

/**
 * @brief Evaluates basis coefficients at sampling points (double to complex version).
 *
 * Transforms basis coefficients to values at sampling points, where input is real
 * (double precision) and output is complex (double precision) values. The operation
 * can be performed along any dimension of a multidimensional array.
 *
 * @param s Pointer to the sampling object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param target_dim Target dimension for the transformation (0-based)
 * @param input Input array of real basis coefficients
 * @param out Output array for the evaluated complex values at sampling points
 *
 * @return 0 on success, non-zero on failure
 *
 * @note For optimal performance, the target dimension should be either the first (0)
 *       or the last (ndim-1) dimension to avoid large temporary array allocations
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note Complex numbers are stored as pairs of consecutive double values (real, imag)
 *
 * @see spir_sampling_evaluate_dd
 * @see spir_sampling_evaluate_zz
 */
int spir_sampling_evaluate_dz(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const double *input,
    std::complex<double> *out)
{
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, out,
                        &sparseir::AbstractSampling::evaluate_inplace_dz);
}

int spir_sampling_evaluate_zz(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const std::complex<double> *input,
    std::complex<double> *out)
{
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, out,
                        &sparseir::AbstractSampling::evaluate_inplace_zz);
}

int spir_sampling_fit_dd(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const double *input,
    double *out)
{
    return fit_impl(s, order, ndim, input_dims, target_dim, input, out,
                        &sparseir::AbstractSampling::fit_inplace_dd);
}

int spir_sampling_fit_zz(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const std::complex<double> *input,
    std::complex<double> *out)
{
    return fit_impl(s, order, ndim, input_dims, target_dim, input, out,
                        &sparseir::AbstractSampling::fit_inplace_zz);
}

/**
 * @brief Gets the number of rows in the fitting matrix of a bosonic DLR.
 *
 * This function returns the number of rows in the fitting matrix of the specified
 * bosonic Discrete Lehmann Representation (DLR). The fitting matrix is used to
 * transform between the DLR representation and values at sampling points.
 *
 * @param dlr Pointer to the bosonic DLR object
 * @return The number of rows in the fitting matrix, or 0 if the DLR object is invalid
 */
size_t spir_bosonic_dlr_fitmat_rows(const spir_bosonic_dlr *dlr)
{
    auto impl = get_impl_bosonic_dlr(dlr);
    if (!impl)
        return 0;
    return impl->fitmat.rows();
}

/**
 * @brief Gets the number of columns in the fitting matrix of a bosonic DLR.
 *
 * This function returns the number of columns in the fitting matrix of the specified
 * bosonic Discrete Lehmann Representation (DLR). The fitting matrix is used to
 * transform between the DLR representation and values at sampling points.
 *
 * @param dlr Pointer to the bosonic DLR object
 * @return The number of columns in the fitting matrix, or 0 if the DLR object is invalid
 */
size_t spir_bosonic_dlr_fitmat_cols(const spir_bosonic_dlr *dlr)
{
    auto impl = get_impl_bosonic_dlr(dlr);
    if (!impl)
        return 0;
    return impl->fitmat.cols();
}

/**
 * @brief Gets the number of rows in the fitting matrix of a fermionic DLR.
 *
 * This function returns the number of rows in the fitting matrix of the specified
 * fermionic Discrete Lehmann Representation (DLR). The fitting matrix is used to
 * transform between the DLR representation and values at sampling points.
 *
 * @param dlr Pointer to the fermionic DLR object
 * @return The number of rows in the fitting matrix, or 0 if the DLR object is invalid
 */
size_t spir_fermionic_dlr_fitmat_rows(const spir_fermionic_dlr *dlr)
{
    auto impl = get_impl_fermionic_dlr(dlr);
    if (!impl)
        return 0;
    return impl->fitmat.rows();
}

/**
 * @brief Gets the number of columns in the fitting matrix of a fermionic DLR.
 *
 * This function returns the number of columns in the fitting matrix of the specified
 * fermionic Discrete Lehmann Representation (DLR). The fitting matrix is used to
 * transform between the DLR representation and values at sampling points.
 *
 * @param dlr Pointer to the fermionic DLR object
 * @return The number of columns in the fitting matrix, or 0 if the DLR object is invalid
 */
size_t spir_fermionic_dlr_fitmat_cols(const spir_fermionic_dlr *dlr)
{
    auto impl = get_impl_fermionic_dlr(dlr);
    if (!impl)
        return 0;
    return impl->fitmat.cols();
}

/**
 * @brief Transforms coefficients from IR basis to bosonic DLR representation.
 *
 * This function converts expansion coefficients from the Intermediate
 * Representation (IR) basis to the Discrete Lehmann Representation (DLR).
 * The transformation is performed by solving a linear system using the
 * fitting matrix:
 *
 * g_DLR = fitmat \ g_IR
 *
 * where:
 * - g_DLR are the coefficients in the DLR basis
 * - g_IR are the coefficients in the IR basis
 * - fitmat is the transformation matrix
 *
 * @param dlr Pointer to the bosonic DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param input Input array of IR coefficients (double precision)
 * @param out Output array for the DLR coefficients (double precision)
 *
 * @return 0 on success, -1 on failure (if the DLR object is invalid or an error occurs)
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note This function is specifically for bosonic (symmetric) Green's functions
 * @note The transformation preserves the numerical properties of the representation
 * @note The transformation involves solving a linear system, which may be
 *       computationally more intensive than the forward transformation
 *
 * @see spir_bosonic_dlr_to_IR
 * @see spir_fermionic_dlr_from_IR
 */
int spir_bosonic_dlr_to_IR(
    const spir_bosonic_dlr *dlr,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    const double *input,
    double *out)
{
    auto impl = get_impl_bosonic_dlr(dlr);
    if (!impl)
        return -1;
    std::array<int32_t, 2> input_dims_2d = collapse_to_2d(ndim, input_dims, 0);
    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::reverse(input_dims_2d.begin(), input_dims_2d.end());
    }
    Eigen::Tensor<double, 2> input_tensor(input_dims_2d[0], input_dims_2d[1]);
    size_t total_input_size = input_dims_2d[0] * input_dims_2d[1];
    for (size_t i = 0; i < total_input_size; i++) {
        input_tensor.data()[i] = input[i];
    }
    Eigen::Tensor<double, 2> out_tensor = impl->to_IR(input_tensor);
    size_t total_output_size = out_tensor.dimension(0) * out_tensor.dimension(1);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor.data()[i];
    }
    return 0;
}

/**
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
 * @return 0 on success, -1 on failure (if the DLR object is invalid or an error
 * occurs)
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
    double *out)
{
    auto impl = get_impl_bosonic_dlr(dlr);
    if (!impl)
        return -1;
    std::array<int32_t, 2> input_dims_2d = collapse_to_2d(ndim, input_dims, 0);
    Eigen::Tensor<double, 2> input_tensor(input_dims_2d[0], input_dims_2d[1]);
    std::size_t total_input_size = input_dims_2d[0] * input_dims_2d[1];
    for (std::size_t i = 0; i < total_input_size; i++) {
        input_tensor.data()[i] = input[i];
    }
    Eigen::Tensor<double, 2> out_tensor = impl->from_IR(input_tensor);
    // pass data to out
    std::size_t total_output_size = out_tensor.dimension(0) * out_tensor.dimension(1);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor.data()[i];
    }
    return 0;
}

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
 * @return 0 on success, -1 on failure (if the DLR object is invalid or an error
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
int spir_fermionic_dlr_to_IR(
    const spir_fermionic_dlr *dlr,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    const double *input,
    double *out)
{
    auto impl = get_impl_fermionic_dlr(dlr);
    if (!impl)
        return -1;
    std::array<int32_t, 2> input_dims_2d = collapse_to_2d(ndim, input_dims, 0);
    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::reverse(input_dims_2d.begin(), input_dims_2d.end());
    }
    Eigen::Tensor<double, 2> input_tensor(input_dims_2d[0], input_dims_2d[1]);
    size_t total_input_size = input_dims_2d[0] * input_dims_2d[1];
    for (size_t i = 0; i < total_input_size; i++) {
        input_tensor.data()[i] = input[i];
    }
    Eigen::Tensor<double, 2> out_tensor = impl->to_IR(input_tensor);
    size_t total_output_size = out_tensor.dimension(0) * out_tensor.dimension(1);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor.data()[i];
    }
    return 0;
}

/**
 * @brief Transforms coefficients from IR basis to fermionic DLR representation.
 *
 * This function converts expansion coefficients from the Intermediate
 * Representation (IR) basis to the Discrete Lehmann Representation (DLR).
 * The transformation is performed by solving a linear system using the
 * SVD-factorized matrix:
 *
 * g_DLR = matrix \ g_IR
 *
 * where:
 * - g_DLR are the coefficients in the DLR basis
 * - g_IR are the coefficients in the IR basis
 * - matrix is the SVD-factorized transformation matrix
 *
 * @param dlr Pointer to the fermionic DLR object
 * @param order Memory layout order (SPIR_ORDER_ROW_MAJOR or
 * SPIR_ORDER_COLUMN_MAJOR)
 * @param ndim Number of dimensions in the input/output arrays
 * @param input_dims Array of dimension sizes
 * @param input Input array of IR coefficients (double precision)
 * @param out Output array for the DLR coefficients (double precision)
 *
 * @return 0 on success, -1 on failure (if the DLR object is invalid or an error
 * occurs)
 *
 * @note The output array must be pre-allocated with the correct size
 * @note The input and output arrays must be contiguous in memory
 * @note This function is specifically for fermionic Green's functions
 * @note The transformation preserves the numerical properties of the
 * representation
 * @note The transformation involves solving a linear system, which may be
 *       computationally more intensive than the forward transformation
 *
 * @see spir_fermionic_dlr_to_IR
 * @see spir_bosonic_dlr_from_IR
 */
int spir_fermionic_dlr_from_IR(
    const spir_fermionic_dlr *dlr,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    const double *input,
    double *out)
{
    auto impl = get_impl_fermionic_dlr(dlr);
    if (!impl)
        return -1;
    std::array<int32_t, 2> input_dims_2d = collapse_to_2d(ndim, input_dims, 0);
    Eigen::Tensor<double, 2> input_tensor(input_dims_2d[0], input_dims_2d[1]);
    std::size_t total_input_size = input_dims_2d[0] * input_dims_2d[1];
    for (std::size_t i = 0; i < total_input_size; i++) {
        input_tensor.data()[i] = input[i];
    }
    Eigen::Tensor<double, 2> out_tensor = impl->from_IR(input_tensor);
    // pass data to out
    std::size_t total_output_size = out_tensor.dimension(0) * out_tensor.dimension(1);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor.data()[i];
    }
    return 0;
}

// Get basis functions (returns the PiecewiseLegendrePolyVector)
/**
 * @brief Retrieves the basis functions in imaginary time from a fermionic finite temperature basis.
 *
 * This function returns the piecewise Legendre polynomial representation of the
 * basis functions u_l(τ) in imaginary time from the specified fermionic finite
 * temperature basis. These basis functions form an orthonormal system on [0, β].
 *
 * @param b Pointer to the fermionic finite temperature basis object
 * @return A pointer to a polyvector object containing the basis functions,
 *         or NULL if the basis object is invalid
 *
 * @note The returned object must be freed using spir_destroy_polyvector when no longer needed
 * @see spir_destroy_polyvector
 */
spir_polyvector *spir_basis_u(const spir_fermionic_finite_temp_basis *b)
{
    auto impl = get_impl_fermionic_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    // Simply use the shared_ptr that already exists in the implementation
    return create_polyvector(impl->u);
}

// Create new regularized bose kernel
// spir_regularized_bosonic_kernel *spir_kernel_regularized_bose_new(double
// lambda)
//{
// try {
// return create_regularized_bose_kernel(
// std::make_shared<sparseir::RegularizedBoseKernel>(lambda));
//} catch (...) {
// return nullptr;
//}
//}

} // extern "C"

// Implementation of the LogisticKernel
namespace sparseir {

} // namespace sparseir
