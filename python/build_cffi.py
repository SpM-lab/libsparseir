from cffi import FFI
import os
import shutil

ffi = FFI()

# Define the C interface
ffi.cdef("""
    typedef double _Complex c_complex;

    typedef enum {
        SPIR_STATISTICS_FERMIONIC = 1,
        SPIR_STATISTICS_BOSONIC = 0
    } spir_statistics_type;

    typedef enum {
        SPIR_ORDER_COLUMN_MAJOR = 1,
        SPIR_ORDER_ROW_MAJOR = 0
    } spir_order_type;

    // Opaque types
    typedef struct _spir_kernel spir_kernel;
    typedef struct _spir_logistic_kernel spir_logistic_kernel;
    typedef struct _spir_regularized_bose_kernel spir_regularized_bose_kernel;
    typedef struct _spir_continuous_functions spir_continuous_functions;
    typedef struct _spir_matsubara_functions spir_matsubara_functions;
    typedef struct _spir_sampling spir_sampling;
    typedef struct _spir_fermionic_finite_temp_basis spir_fermionic_finite_temp_basis;
    typedef struct _spir_bosonic_finite_temp_basis spir_bosonic_finite_temp_basis;
    typedef struct _spir_fermionic_dlr spir_fermionic_dlr;
    typedef struct _spir_bosonic_dlr spir_bosonic_dlr;

    // Core functions
    void spir_destroy_kernel(spir_kernel *obj);
    spir_kernel *spir_clone_kernel(const spir_kernel *src);
    int spir_is_assigned_kernel(const spir_kernel *obj);
    spir_kernel *spir_logistic_kernel_new(double lambda);
    spir_kernel *spir_regularized_bose_kernel_new(double lambda);

    int spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                          double *ymin, double *ymax);
    int spir_kernel_matrix(const spir_kernel *k, const double *x, int nx,
                          const double *y, int ny, double *out);

    // Basis functions
    spir_fermionic_finite_temp_basis* spir_fermionic_finite_temp_basis_new(
        double beta, double omega_max, double epsilon);
    spir_bosonic_finite_temp_basis* spir_bosonic_finite_temp_basis_new(
        double beta, double omega_max, double epsilon);

    // Sampling functions
    int spir_sampling_evaluate_dd(
        const spir_sampling *s,
        spir_order_type order,
        int32_t ndim,
        int32_t *input_dims,
        int32_t target_dim,
        const double *input,
        double *out
    );

    int spir_sampling_evaluate_dz(
        const spir_sampling *s,
        spir_order_type order,
        int32_t ndim,
        int32_t *input_dims,
        int32_t target_dim,
        const double *input,
        c_complex *out
    );

    int spir_sampling_evaluate_zz(
        const spir_sampling *s,
        spir_order_type order,
        int32_t ndim,
        int32_t *input_dims,
        int32_t target_dim,
        const c_complex *input,
        c_complex *out
    );
""")

# Get absolute paths
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "libsparseir"))
include_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "include"))

# Set up the build
ffi.set_source("_libsparseir_cffi",
    """
    #include <complex.h>
    #include "sparseir/sparseir.h"
    """,
    libraries=['sparseir'],
    library_dirs=[build_dir],
    include_dirs=[include_dir],
    extra_link_args=['-Wl,-rpath,' + build_dir],
    extra_compile_args=['-Wno-unused-command-line-argument']
)

if __name__ == "__main__":
    # Build the CFFI module
    ffi.compile()

    # Move the generated files to the package directory
    for filename in os.listdir('.'):
        if filename.startswith('_libsparseir_cffi'):
            src = os.path.join('.', filename)
            dst = os.path.join(package_dir, filename)
            shutil.move(src, dst)