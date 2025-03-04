#include "sparseir/sparseir.h"
#include "sparseir/sparseir.hpp"
#include "sparseir/kernel.hpp"
#include <stdexcept>

// Implementation of the opaque type
struct _spir_kernel {
    // Pointer to the C++ implementation
    std::unique_ptr<sparseir::AbstractKernel> impl;
    
    // Constructor
    explicit _spir_kernel(std::unique_ptr<sparseir::AbstractKernel> k) 
        : impl(std::move(k)) {}
};

// Implementation of the opaque type
struct _spir_basis {
    // Pointer to the C++ implementation
    std::unique_ptr<sparseir::FiniteTempBasis<sparseir::Fermionic>> impl;
    
    // Constructor
    explicit _spir_basis(std::unique_ptr<sparseir::FiniteTempBasis<sparseir::Fermionic>> b) 
        : impl(std::move(b)) {}
};

// Implementation of the C API
extern "C" {

spir_kernel* spir_logistic_kernel(double lambda) {
    try {
        auto impl = std::make_unique<sparseir::LogisticKernel>(lambda);
        return new _spir_kernel(std::move(impl));
    } catch (...) {
        return nullptr;
    }
}

void spir_destroy_kernel(spir_kernel* k) {
    delete k;
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
spir_basis* spir_basis_new(double beta, double omega_max, double epsilon) {
    try {
        auto impl = std::make_unique<sparseir::FiniteTempBasis<sparseir::Fermionic>>(
            beta, omega_max, epsilon, 
            sparseir::LogisticKernel(beta * omega_max)
        );
        return new _spir_basis(std::move(impl));
    } catch (...) {
        return nullptr;
    }
}

// Destructor
void spir_destroy_basis(spir_basis* b) {
    delete b;
}

} // extern "C"

// Implementation of the LogisticKernel
namespace sparseir {



} // namespace sparseir
