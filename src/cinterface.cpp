#include "sparseir/sparseir.h"
#include "sparseir/sparseir.hpp"
#include "sparseir/utils.hpp"
#include <memory>
#include <stdexcept>
#include <cstdint>

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
    /* Destroy function */                                                     \
    void spir_destroy_##name(spir_##name *obj)                                 \
    {                                                                          \
        if (obj) {                                                             \
            delete obj;                                                        \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Helper to get the implementation shared_ptr */                          \
    static inline std::shared_ptr<impl_type> get_impl_##name(const spir_##name *obj) \
    {                                                                          \
        if (!obj)                                                              \
            return nullptr;                                                    \
        return obj->ptr;                                                       \
    }

// Implementation of the opaque types
IMPLEMENT_OPAQUE_TYPE(kernel, sparseir::AbstractKernel);
IMPLEMENT_OPAQUE_TYPE(logistic_kernel, sparseir::LogisticKernel);
IMPLEMENT_OPAQUE_TYPE(polyvector, sparseir::PiecewiseLegendrePolyVector);
IMPLEMENT_OPAQUE_TYPE(basis, sparseir::FiniteTempBasis<sparseir::Fermionic>);
IMPLEMENT_OPAQUE_TYPE(fermionic_finite_temp_basis,
                      sparseir::FiniteTempBasis<sparseir::Fermionic>);
IMPLEMENT_OPAQUE_TYPE(sampling, sparseir::AbstractSampling);
IMPLEMENT_OPAQUE_TYPE(sve_result, sparseir::SVEResult);

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

// Create new logistic kernel
spir_kernel *spir_logistic_kernel_new(double lambda)
{
    try {
        auto kernel = std::make_shared<sparseir::LogisticKernel>(lambda);
        auto abstract_kernel = std::shared_ptr<sparseir::AbstractKernel>(kernel);
        return create_kernel(abstract_kernel);
    } catch (...) {
        return nullptr;
    }
}

int spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                       double *ymin, double *ymax)
{
    auto impl = get_impl_kernel(k);
    if (!impl)
        return -1;

    try {
        auto xrange = impl->xrange();
        auto yrange = impl->yrange();

        *xmin = xrange.first;
        *xmax = xrange.second;
        *ymin = yrange.first;
        *ymax = yrange.second;

        return 0;
    } catch (...) {
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

// Create new SVE result
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

// Constructor for basis
spir_fermionic_finite_temp_basis *
spir_fermionic_finite_temp_basis_new(double beta, double omega_max,
                                     double epsilon)
{
    try {
        return create_fermionic_finite_temp_basis(
            std::make_shared<sparseir::FiniteTempBasis<sparseir::Fermionic>>(
                beta, omega_max, epsilon,
                sparseir::LogisticKernel(beta * omega_max)));
    } catch (...) {
        return nullptr;
    }
}

spir_sampling *
spir_fermionic_tau_sampling_new(const spir_fermionic_finite_temp_basis *b)
{
    auto impl = get_impl_fermionic_finite_temp_basis(b);
    if (!impl)
        return nullptr;
    auto smpl = std::make_shared<sparseir::TauSampling<sparseir::Fermionic>>(*impl);
    return create_sampling(smpl);
}

spir_sampling *
spir_fermionic_matsubara_sampling_new(const spir_fermionic_finite_temp_basis *b)
{
    auto impl = get_impl_fermionic_finite_temp_basis(b);
    if (!impl)
        return nullptr;
    auto smpl = std::make_shared<sparseir::MatsubaraSampling<sparseir::Fermionic>>(*impl);
    return create_sampling(smpl);
}

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

int spir_sampling_evaluate_dc(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const double *input,
    std::complex<double> *out)
{
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, out,
                        &sparseir::AbstractSampling::evaluate_inplace_dc);
}

int spir_sampling_evaluate_cc(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const std::complex<double> *input,
    std::complex<double> *out)
{
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, out,
                        &sparseir::AbstractSampling::evaluate_inplace_cc);
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

int spir_sampling_fit_cc(
    const spir_sampling *s,
    spir_order_type order,
    int32_t ndim,
    int32_t *input_dims,
    int32_t target_dim,
    const std::complex<double> *input,
    std::complex<double> *out)
{
    return fit_impl(s, order, ndim, input_dims, target_dim, input, out,
                        &sparseir::AbstractSampling::fit_inplace_cc);
}


// Get basis functions (returns the PiecewiseLegendrePolyVector)
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
