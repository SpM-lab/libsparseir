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

int spir_sampling_evaluate_dd(
    const spir_sampling *s,        // Sampling object
    spir_order_type order,         // Order type (C or Fortran)
    int32_t ndim,                  // Number of dimensions
    int32_t *input_dims,                 // Array of dimensions
    int32_t target_dim,            // Target dimension for evaluation
    const double *input,          // Input coefficients array
    double *out                    // Output array
    )
{
    auto impl = get_impl_sampling(s);
    if (!impl)
        return -1;
    
    // Convert N-dimensional array to 3D array by collapsing dimensions before and after target_dim
    std::array<int32_t, 3> in_dims_3d = {1, input_dims[target_dim], 1};
    // Multiply all dimensions before target_dim into first dimension
    for (int32_t i = 0; i < target_dim; ++i) {
        in_dims_3d[0] *= input_dims[i];
    }
    // Multiply all dimensions after target_dim into last dimension 
    for (int32_t i = target_dim + 1; i < ndim; ++i) {
        in_dims_3d[2] *= input_dims[i];
    }
    // Output dimensions match input dimensions after collapsing to 3D
    std::array<int32_t, 3> out_dims_3d = {in_dims_3d[0], input_dims[target_dim], in_dims_3d[2]};

    // Create TensorMap for input coefficients (3D)
    Eigen::TensorMap<const Eigen::Tensor<double, 3>> input_3d(input, in_dims_3d);

    // Create TensorMap for output (3D)
    Eigen::TensorMap<Eigen::Tensor<double, 3>> output_3d(out, out_dims_3d);

    if (order == SPIR_ORDER_ROW_MAJOR) { // C-style order
        // Eigen3 uses column-major order by default. Convert to row-major order.
        // Transpose input_3d to Fortran-style order
        input_3d = input_3d.shuffle(std::array<int32_t, 3>({2, 1, 0}));
        output_3d = output_3d.shuffle(std::array<int32_t, 3>({2, 1, 0}));
    }

    int err = impl->evaluate_inplace_dd(input_3d, 1, output_3d);

    return err;
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
