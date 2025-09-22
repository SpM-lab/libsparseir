#include "sparseir/gemm.hpp"

namespace sparseir {

#ifdef SPARSEIR_USE_BLAS

#ifdef SPARSEIR_USE_EXTERN_FBLAS_PTR

#include <cstddef>
#include <stdexcept>

// Fortran dgemm 関数ポインタ型
using dgemm_fptr = void(*)(const char*, const char*, const int*,
                           const int*, const int*, const double*,
                           const double*, const int*, const double*,
                           const int*, const double*, double*, const int*);

using zgemm_fptr = void(*)(const char*, const char*, const int*,
                           const int*, const int*, const void*,
                           const void*, const int*, const void*,
                           const int*, const void*, void*, const int*);

// 保存場所（最初は未登録）
static dgemm_fptr registered_dgemm = nullptr;

static zgemm_fptr registered_zgemm = nullptr;

extern "C" {


// Python 側から関数ポインタを登録する
void spir_register_dgemm(void* fn) {
    registered_dgemm = reinterpret_cast<dgemm_fptr>(fn);
}

void spir_register_zgemm(void* fn) {
    registered_zgemm = reinterpret_cast<zgemm_fptr>(fn);
}

} // extern "C"

#endif

// Implementation of my_dgemm - converts CBLAS interface to Fortran BLAS
void my_dgemm(const int Order, const int TransA, const int TransB,
              const int64_t M, const int64_t N, const int64_t K, const double alpha,
              const double *A, const int64_t lda, const double *B, const int64_t ldb,
              const double beta, double *C, const int64_t ldc)
{
    // Convert CBLAS transpose flags to Fortran character codes
    char transa = (TransA == CblasNoTrans) ? 'N' : (TransA == CblasTrans) ? 'T' : 'C';
    char transb = (TransB == CblasNoTrans) ? 'N' : (TransB == CblasTrans) ? 'T' : 'C';

#ifdef SPARSEIR_USE_EXTERN_FBLAS_PTR
    if (!registered_dgemm) {
        throw std::runtime_error("dgemm not registered (call spir_register_dgemm from Python first)");
    }
    // Convert int64_t to int for Fortran interface
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda_int = static_cast<int>(lda), ldb_int = static_cast<int>(ldb), ldc_int = static_cast<int>(ldc);
    registered_dgemm(&transa, &transb, &m, &n, &k, &alpha, A, &lda_int, B, &ldb_int, &beta, C, &ldc_int);
#else
#ifdef SPARSEIR_USE_ILP64
    // ILP64: Convert to long long for Fortran interface
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda_ll = static_cast<long long>(lda), ldb_ll = static_cast<long long>(ldb), ldc_ll = static_cast<long long>(ldc);
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda_ll, B, &ldb_ll, &beta, C, &ldc_ll);
#else
    // LP64: Convert to int for Fortran interface
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda_int = static_cast<int>(lda), ldb_int = static_cast<int>(ldb), ldc_int = static_cast<int>(ldc);
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda_int, B, &ldb_int, &beta, C, &ldc_int);
#endif
#endif // SPARSEIR_USE_EXTERN_FBLAS_PTR
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

#ifdef SPARSEIR_USE_EXTERN_FBLAS_PTR
    if (!registered_zgemm) {
        throw std::runtime_error("zgemm not registered (call spir_register_zgemm from Python first)");
    }
    // Convert int64_t to int for Fortran interface
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda_int = static_cast<int>(lda), ldb_int = static_cast<int>(ldb), ldc_int = static_cast<int>(ldc);
    registered_zgemm(&transa, &transb, &m, &n, &k, alpha, A, &lda_int, B, &ldb_int, beta, C, &ldc_int);
#else
#ifdef SPARSEIR_USE_ILP64
    // ILP64: Convert to long long for Fortran interface
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda_ll = static_cast<long long>(lda), ldb_ll = static_cast<long long>(ldb), ldc_ll = static_cast<long long>(ldc);
    zgemm_(&transa, &transb, &m, &n, &k, alpha, A, &lda_ll, B, &ldb_ll, beta, C, &ldc_ll);
#else
    // LP64: Convert to int for Fortran interface
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda_int = static_cast<int>(lda), ldb_int = static_cast<int>(ldb), ldc_int = static_cast<int>(ldc);
    zgemm_(&transa, &transb, &m, &n, &k, alpha, A, &lda_int, B, &ldb_int, beta, C, &ldc_int);
#endif
#endif // SPARSEIR_USE_EXTERN_FBLAS_PTR
}

#endif // SPARSEIR_USE_BLAS

} // namespace sparseir
