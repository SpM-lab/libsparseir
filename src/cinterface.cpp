#include "sparseir/sparseir.h"
#include "sparseir/sparseir.hpp"
#include "sparseir/kernel.hpp"
#include <stdexcept>

// Define opaque type and implement its management functions
#define IMPLEMENT_OPAQUE_TYPE(name, impl_type) \
struct _spir_##name { \
    impl_type* impl;  /* Pointer to implementation */ \
    bool owns;        /* true if this object owns the implementation */ \
}; \
typedef struct _spir_##name spir_##name; \
\
/* Helper functions for creating objects */ \
static inline spir_##name* create_owned_##name(std::unique_ptr<impl_type> p) { \
    auto* obj = new spir_##name; \
    obj->impl = p.release(); \
    obj->owns = true; \
    return obj; \
} \
\
static inline spir_##name* create_view_##name(impl_type* p) { \
    auto* obj = new spir_##name; \
    obj->impl = p; \
    obj->owns = false; \
    return obj; \
} \
\
/* Destroy function */ \
void spir_destroy_##name(spir_##name* obj) { \
    if (obj) { \
        if (obj->owns) delete obj->impl; \
        delete obj; \
    } \
}

// Implementation of the opaque types
IMPLEMENT_OPAQUE_TYPE(kernel, sparseir::AbstractKernel);
IMPLEMENT_OPAQUE_TYPE(polyvector, sparseir::PiecewiseLegendrePolyVector);
IMPLEMENT_OPAQUE_TYPE(basis, sparseir::FiniteTempBasis<sparseir::Fermionic>);
IMPLEMENT_OPAQUE_TYPE(fermionic_basis, sparseir::FiniteTempBasis<sparseir::Fermionic>);


// Implementation of the C API
extern "C" {

// Create new logistic kernel
spir_kernel* spir_kernel_new_logistic(double lambda) {
    try {
        return create_owned_kernel(std::make_unique<sparseir::LogisticKernel>(lambda));
    } catch (...) {
        return nullptr;
    }
}

int spir_kernel_domain(const spir_kernel* k, 
                      double* xmin, double* xmax,
                      double* ymin, double* ymax) {
    if (!k || !k->impl) return -1;
    
    try {
        auto xrange = k->impl->xrange();
        auto yrange = k->impl->yrange();
        
        *xmin = xrange.first;
        *xmax = xrange.second;
        *ymin = yrange.first;
        *ymax = yrange.second;
        
        return 0;
    } catch (...) {
        return -1;
    }
}

int spir_kernel_evaluate(const spir_kernel* k, double x, double y, double* out) {
    if (!k || !k->impl || !out) return -1;
    
    try {
        *out = k->impl->compute(x, y);
        return 0;
    } catch (...) {
        return -1;
    }
}


int spir_kernel_matrix(const spir_kernel* k, 
                      const double* x, int nx,
                      const double* y, int ny, 
                      double* out) {
    if (!k || !k->impl || !x || !y || !out) return -1;
    if (nx <= 0 || ny <= 0) return -1;
    
    try {
        // Evaluate kernel at each point
        for (int ix = 0; ix < nx; ++ix) {
            for (int iy = 0; iy < ny; ++iy) {
                out[ix * ny + iy] = k->impl->compute(x[ix], y[iy]); // column-major order
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

// Constructor for basis
spir_fermionic_basis* spir_fermionic_basis_new(double beta, double omega_max, double epsilon) {
    try {
        return create_owned_fermionic_basis(
            std::make_unique<sparseir::FiniteTempBasis<sparseir::Fermionic>>(
                beta, omega_max, epsilon, 
                sparseir::LogisticKernel(beta * omega_max)
            )
        );
    } catch (...) {
        return nullptr;
    }
}

// Get basis functions (returns a view of PiecewiseLegendrePolyVector)
spir_polyvector* spir_basis_u_new(const spir_fermionic_basis* b) {
    if (!b || !b->impl) return nullptr;
    
    try {
        // Create a view of the basis functions
        return create_view_polyvector(&b->impl->u);
    } catch (...) {
        return nullptr;
    }
}

// Create new regularized bose kernel
spir_kernel* spir_kernel_new_regularized_bose(double lambda) {
    try {
        return create_owned_kernel(std::make_unique<sparseir::RegularizedBoseKernel>(lambda));
    } catch (...) {
        return nullptr;
    }
}

} // extern "C"

// Implementation of the LogisticKernel
namespace sparseir {



} // namespace sparseir
