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
    typedef struct _spir_continuous_functions spir_continuous_functions;
    typedef struct _spir_matsubara_functions spir_matsubara_functions;
    typedef struct _spir_fermionic_finite_temp_basis spir_fermionic_finite_temp_basis;
    typedef struct _spir_bosonic_finite_temp_basis spir_bosonic_finite_temp_basis;
    typedef struct _spir_sampling spir_sampling;
    typedef struct _spir_sve_result spir_sve_result;
    typedef struct _spir_fermionic_dlr spir_fermionic_dlr;
    typedef struct _spir_bosonic_dlr spir_bosonic_dlr;

    // Kernel functions
    void spir_destroy_kernel(spir_kernel *obj);
    spir_kernel *spir_clone_kernel(const spir_kernel *src);
    int spir_is_assigned_kernel(const spir_kernel *obj);
    spir_kernel *spir_logistic_kernel_new(double lambda);
    spir_kernel *spir_regularized_bose_kernel_new(double lambda);
    int spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                          double *ymin, double *ymax);
    int spir_kernel_matrix(const spir_kernel *k, const double *x, int nx,
                          const double *y, int ny, double *out);

    // SVE functions
    spir_sve_result* spir_sve_result_new(const spir_kernel* k, double epsilon);
    void spir_destroy_sve_result(spir_sve_result *obj);
    spir_sve_result *spir_clone_sve_result(const spir_sve_result *src);
    int spir_is_assigned_sve_result(const spir_sve_result *obj);

    // Basis functions
    spir_fermionic_finite_temp_basis* spir_fermionic_finite_temp_basis_new(
        double beta, double omega_max, double epsilon);
    spir_bosonic_finite_temp_basis* spir_bosonic_finite_temp_basis_new(
        double beta, double omega_max, double epsilon);
    spir_fermionic_finite_temp_basis* spir_fermionic_finite_temp_basis_new_with_sve(
        double beta, double omega_max, const spir_kernel *k, const spir_sve_result *sve);
    spir_bosonic_finite_temp_basis* spir_bosonic_finite_temp_basis_new_with_sve(
        double beta, double omega_max, const spir_kernel *k, const spir_sve_result *sve);
    void spir_destroy_fermionic_finite_temp_basis(spir_fermionic_finite_temp_basis *obj);
    void spir_destroy_bosonic_finite_temp_basis(spir_bosonic_finite_temp_basis *obj);
    int spir_fermionic_finite_temp_basis_get_size(const spir_fermionic_finite_temp_basis *b, int *size);
    int spir_bosonic_finite_temp_basis_get_size(const spir_bosonic_finite_temp_basis *b, int *size);

    // Basis function access
    spir_continuous_functions* spir_fermionic_finite_temp_basis_get_u(const spir_fermionic_finite_temp_basis* b);
    spir_continuous_functions* spir_fermionic_finite_temp_basis_get_v(const spir_fermionic_finite_temp_basis* b);
    spir_matsubara_functions* spir_fermionic_finite_temp_basis_get_uhat(const spir_fermionic_finite_temp_basis* b);
    spir_continuous_functions* spir_bosonic_finite_temp_basis_get_u(const spir_bosonic_finite_temp_basis* b);
    spir_continuous_functions* spir_bosonic_finite_temp_basis_get_v(const spir_bosonic_finite_temp_basis* b);
    spir_matsubara_functions* spir_bosonic_finite_temp_basis_get_uhat(const spir_bosonic_finite_temp_basis* b);

    // Sampling functions
    spir_sampling *spir_fermionic_tau_sampling_new(const spir_fermionic_finite_temp_basis *b);
    spir_sampling *spir_fermionic_matsubara_sampling_new(const spir_fermionic_finite_temp_basis *b);
    spir_sampling *spir_bosonic_tau_sampling_new(const spir_bosonic_finite_temp_basis *b);
    spir_sampling *spir_bosonic_matsubara_sampling_new(const spir_bosonic_finite_temp_basis *b);
    void spir_destroy_sampling(spir_sampling *obj);
    spir_sampling *spir_clone_sampling(const spir_sampling *src);
    int spir_is_assigned_sampling(const spir_sampling *obj);

    // Sampling evaluation functions
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

    // Sampling fitting functions
    int spir_sampling_fit_dd(
        const spir_sampling *s,
        spir_order_type order,
        int32_t ndim,
        int32_t *input_dims,
        int32_t target_dim,
        const double *input,
        double *out
    );

    int spir_sampling_fit_zz(
        const spir_sampling *s,
        spir_order_type order,
        int32_t ndim,
        int32_t *input_dims,
        int32_t target_dim,
        const c_complex *input,
        c_complex *out
    );

    // Sampling point access
    int spir_sampling_get_num_points(const spir_sampling *s, int *num_points);
    int spir_sampling_get_tau_points(const spir_sampling *s, double *points);
    int spir_sampling_get_matsubara_points(const spir_sampling *s, int *points);

    // DLR functions
    spir_fermionic_dlr *spir_fermionic_dlr_new(const spir_fermionic_finite_temp_basis *b);
    spir_fermionic_dlr *spir_fermionic_dlr_new_with_poles(
        const spir_fermionic_finite_temp_basis *b, const int npoles, const double *poles);
    spir_bosonic_dlr *spir_bosonic_dlr_new(const spir_bosonic_finite_temp_basis *b);
    spir_bosonic_dlr *spir_bosonic_dlr_new_with_poles(
        const spir_bosonic_finite_temp_basis *b, const int npoles, const double *poles);
    void spir_destroy_fermionic_dlr(spir_fermionic_dlr *obj);
    void spir_destroy_bosonic_dlr(spir_bosonic_dlr *obj);

    // DLR matrix access
    int spir_fermionic_dlr_fitmat_rows(const spir_fermionic_dlr *dlr);
    int spir_fermionic_dlr_fitmat_cols(const spir_fermionic_dlr *dlr);
    int spir_bosonic_dlr_fitmat_rows(const spir_bosonic_dlr *dlr);
    int spir_bosonic_dlr_fitmat_cols(const spir_bosonic_dlr *dlr);

    // DLR transformation functions
    int spir_fermionic_dlr_from_IR(
        const spir_fermionic_dlr *dlr,
        spir_order_type order,
        int32_t ndim,
        int32_t *input_dims,
        const double *input,
        double *out
    );

    int spir_bosonic_dlr_from_IR(
        const spir_bosonic_dlr *dlr,
        spir_order_type order,
        int32_t ndim,
        int32_t *input_dims,
        const double *input,
        double *out
    );

    int spir_fermionic_dlr_to_IR(
        const spir_fermionic_dlr *dlr,
        spir_order_type order,
        int32_t ndim,
        int32_t *input_dims,
        const double *input,
        double *out
    );

    int spir_bosonic_dlr_to_IR(
        const spir_bosonic_dlr *dlr,
        spir_order_type order,
        int32_t ndim,
        int32_t *input_dims,
        const double *input,
        double *out
    );

    // Basis function evaluation
    int32_t spir_evaluate_continuous_functions(
        const spir_continuous_functions* uv,
        double x,
        double* out
    );

    int32_t spir_evaluate_matsubara_functions(
        const spir_matsubara_functions* uiw,
        spir_order_type order,
        int32_t num_freqs,
        int32_t* matsubara_freq_indices,
        c_complex* out
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