#include "sparseir/sparseir.h"
#include "sparseir/sparseir.hpp"
#include "sparseir/utils.hpp"
#include <memory>
#include <stdexcept>

// Define opaque type and implement its management functions
#define IMPLEMENT_OPAQUE_TYPE(name, impl_type)                                 \
    struct _spir_##name                                                        \
    {                                                                          \
        enum PtrType { Unique, Shared };                                       \
        PtrType type;                                                          \
        union {                                                                \
            std::unique_ptr<impl_type> unique;                                 \
            std::shared_ptr<impl_type> shared;                                 \
        };                                                                     \
        _spir_##name() : type(Unique) { }                                      \
        ~_spir_##name()                                                        \
        {                                                                      \
            if (type == Unique)                                                \
                unique.~unique_ptr<impl_type>();                               \
            else                                                               \
                shared.~shared_ptr<impl_type>();                               \
        }                                                                      \
    };                                                                         \
    typedef struct _spir_##name spir_##name;                                   \
                                                                               \
    /* Helper functions for creating objects */                                \
    static inline spir_##name *create_owned_##name(                            \
        std::unique_ptr<impl_type> p)                                          \
    {                                                                          \
        auto *obj = new spir_##name;                                           \
        obj->type = _spir_##name::Unique;                                      \
        new (&obj->unique) std::unique_ptr<impl_type>(std::move(p));           \
        return obj;                                                            \
    }                                                                          \
                                                                               \
    static inline spir_##name *create_view_##name(                             \
        std::shared_ptr<impl_type> p)                                          \
    {                                                                          \
        auto *obj = new spir_##name;                                           \
        obj->type = _spir_##name::Shared;                                      \
        new (&obj->shared) std::shared_ptr<impl_type>(std::move(p));           \
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
    /* Helper to get the implementation pointer */                             \
    static inline impl_type *get_impl_##name(const spir_##name *obj)           \
    {                                                                          \
        if (!obj)                                                              \
            return nullptr;                                                    \
        if (obj->type == _spir_##name::Unique) {                               \
            return obj->unique.get();                                          \
        } else {                                                               \
            return obj->shared.get();                                          \
        }                                                                      \
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
        auto kernel =
            sparseir::util::make_unique<sparseir::LogisticKernel>(lambda);
        auto abstract_kernel =
            std::unique_ptr<sparseir::AbstractKernel>(kernel.release());
        return create_owned_kernel(std::move(abstract_kernel));
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
        return create_owned_fermionic_finite_temp_basis(
            sparseir::util::make_unique<
                sparseir::FiniteTempBasis<sparseir::Fermionic>>(
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
    auto smpl =
        sparseir::util::make_unique<sparseir::TauSampling<sparseir::Fermionic>>(
            *impl);
    return create_owned_sampling(std::move(smpl));
}

int spir_sampling_evaluate(const spir_sampling *s, spir_order_type order,
                           const double *coeffs, int n_components,
                           int target_dim, double *out)
{
    auto impl = get_impl_sampling(s);
    if (!impl)
        return -1;
    Eigen::TensorMap<const Eigen::Tensor<double, 2>> in_mat(coeffs, impl->basis_size(), n_components);
    //impl->evaluate(coeffs, n_components, target_dim, out_mat);
    return 0;
}

// Get basis functions (returns a view of PiecewiseLegendrePolyVector)
spir_polyvector *spir_basis_u_view(const spir_fermionic_finite_temp_basis *b)
{
    auto impl = get_impl_fermionic_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    try {
        // Create a view of the basis functions using a non-owning shared_ptr
        // The empty deleter ensures the object is not deleted when the
        // shared_ptr is destroyed
        auto shared_view =
            std::shared_ptr<sparseir::PiecewiseLegendrePolyVector>(
                &impl->u, [](sparseir::PiecewiseLegendrePolyVector *) { });
        return create_view_polyvector(shared_view);
    } catch (...) {
        return nullptr;
    }
}

// Create new regularized bose kernel
// spir_regularized_bosonic_kernel *spir_kernel_regularized_bose_new(double
// lambda)
//{
// try {
// return create_owned_regularized_bose_kernel(
// sparseir::util::make_unique<sparseir::RegularizedBoseKernel>(lambda));
//} catch (...) {
// return nullptr;
//}
//}

} // extern "C"

// Implementation of the LogisticKernel
namespace sparseir {

} // namespace sparseir
