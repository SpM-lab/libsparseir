#include "sparseir/gemm.hpp"

#ifdef SPARSEIR_USE_BLAS
namespace sparseir {


#include <cstddef>
#include <stdexcept>

// Fortran dgemm function pointer type
#ifdef SPARSEIR_USE_ILP64
using dgemm_fptr = void(*)(const char*, const char*, const int64_t*,
                           const int64_t*, const int64_t*, const double*,
                           const double*, const int64_t*, const double*,
                           const int64_t*, const double*, double*, const int64_t*);

using zgemm_fptr = void(*)(const char*, const char*, const int64_t*,
                           const int64_t*, const int64_t*, const void*,
                           const void*, const int64_t*, const void*,
                           const int64_t*, const void*, void*, const int64_t*);
#else
using dgemm_fptr = void(*)(const char*, const char*, const int*,
                           const int32_t*, const int32_t*, const double*,
                           const double*, const int32_t*, const double*,
                           const int32_t*, const double*, double*, const int32_t*);

using zgemm_fptr = void(*)(const char*, const char*, const int32_t*,
                           const int32_t*, const int32_t*, const void*,
                           const void*, const int32_t*, const void*,
                           const int32_t*, const void*, void*, const int32_t*);
#endif


#ifdef SPARSEIR_USE_EXTERN_FBLAS_PTR
// Storage for registered function pointers (initially not registered)
static dgemm_fptr registered_dgemm = nullptr;
static zgemm_fptr registered_zgemm = nullptr;

extern "C" {

// Register function pointers from outside
void spir_register_dgemm(void* fn) {
    registered_dgemm = reinterpret_cast<dgemm_fptr>(fn);
}

void spir_register_zgemm(void* fn) {
    registered_zgemm = reinterpret_cast<zgemm_fptr>(fn);
}

} // extern "C"

#else

static dgemm_fptr registered_dgemm = dgemm_;
static zgemm_fptr registered_zgemm = zgemm_;

#endif // SPARSEIR_USE_EXTERN_FBLAS_PTR

// Implementation of my_dgemm - converts CBLAS interface to Fortran BLAS
void my_dgemm(const int Order, const int TransA, const int TransB,
              const int64_t M, const int64_t N, const int64_t K, const double alpha,
              const double *A, const int64_t lda, const double *B, const int64_t ldb,
              const double beta, double *C, const int64_t ldc)
{
    // Convert CBLAS transpose flags to Fortran character codes
    char transa = (TransA == CblasNoTrans) ? 'N' : (TransA == CblasTrans) ? 'T' : 'C';
    char transb = (TransB == CblasNoTrans) ? 'N' : (TransB == CblasTrans) ? 'T' : 'C';

#ifdef SPARSEIR_USE_ILP64
    // ILP64: Convert to long long for Fortran interface
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda_ll = static_cast<long long>(lda), ldb_ll = static_cast<long long>(ldb), ldc_ll = static_cast<long long>(ldc);
    registered_dgemm(&transa, &transb, &m, &n, &k, &alpha, A, &lda_ll, B, &ldb_ll, &beta, C, &ldc_ll);
#else
    // LP64: Convert to int for Fortran interface
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda_int = static_cast<int>(lda), ldb_int = static_cast<int>(ldb), ldc_int = static_cast<int>(ldc);
    registered_dgemm(&transa, &transb, &m, &n, &k, &alpha, A, &lda_int, B, &ldb_int, &beta, C, &ldc_int);
#endif
}

// Implementation of my_zgemm - converts CBLAS interface to Fortran BLAS
void my_zgemm(const int Order, const int TransA, const int TransB,
              const int64_t M, const int64_t N, const int64_t K, const void *alpha,
              const void *A, const int64_t lda, const void *B, const int64_t ldb,
              const void *beta, void *C, const int64_t ldc)
{
    // Convert CBLAS transpose flags to Fortran character codes
    char transa = (TransA == CblasNoTrans) ? 'N' : (TransA == CblasTrans) ? 'T' : 'C';
    char transb = (TransB == CblasNoTrans) ? 'N' : (TransB == CblasTrans) ? 'T' : 'C';

    if (!registered_zgemm) {
        throw std::runtime_error("zgemm not registered");
    }
#ifdef SPARSEIR_USE_ILP64
    // ILP64: Convert to long long for Fortran interface
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda_ll = static_cast<long long>(lda), ldb_ll = static_cast<long long>(ldb), ldc_ll = static_cast<long long>(ldc);
    registered_zgemm(&transa, &transb, &m, &n, &k, alpha, A, &lda_ll, B, &ldb_ll, beta, C, &ldc_ll);
#else
    // LP64: Convert to int for Fortran interface
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda_int = static_cast<int>(lda), ldb_int = static_cast<int>(ldb), ldc_int = static_cast<int>(ldc);
    registered_zgemm(&transa, &transb, &m, &n, &k, alpha, A, &lda_int, B, &ldb_int, beta, C, &ldc_int);
#endif
}

} // namespace sparseir


#endif // SPARSEIR_USE_BLAS
