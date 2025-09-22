// blas_wrapper.cpp
#include "sparseir/gemm.hpp"

namespace sparseir {

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

// libsparseir 内部が呼び出すラッパー
void my_dgemm(const char* transa, const char* transb,
              const int* m, const int* n, const int* k,
              const double* alpha,
              const double* a, const int* lda,
              const double* b, const int* ldb,
              const double* beta,
              double* c, const int* ldc) {
    if (!registered_dgemm) {
        throw std::runtime_error("dgemm not registered (call spir_register_dgemm from Python first)");
    }
    registered_dgemm(transa, transb, m, n, k,
                     alpha, a, lda, b, ldb, beta, c, ldc);
}

void my_zgemm(const char* transa, const char* transb,
              const int* m, const int* n, const int* k,
              const void* alpha,
              const void* a, const int* lda,
              const void* b, const int* ldb,
              const void* beta, void* c, const int* ldc) {
    if (!registered_zgemm) {
        throw std::runtime_error("zgemm not registered (call spir_register_zgemm from Python first)");
    }
    registered_zgemm(transa, transb, m, n, k,
                     alpha, a, lda, b, ldb, beta, c, ldc);
}

} // extern "C"

} // namespace sparseir