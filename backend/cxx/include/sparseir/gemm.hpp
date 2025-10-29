#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>
#include <complex>
#include <tuple>
#include <cstdint>

#include "sparseir/jacobi_svd.hpp"

namespace sparseir {

// Fortran BLAS interface wrapper functions
// Integer type is selected at compile time based on SPARSEIR_USE_BLAS_ILP64
#ifdef SPARSEIR_USE_BLAS_ILP64
// ILP64 interface: 64-bit integers
void my_dgemm(const char* transa, const char* transb,
              const long long* m, const long long* n, const long long* k,
              const double* alpha, const double* a, const long long* lda,
              const double* b, const long long* ldb, const double* beta,
              double* c, const long long* ldc);

void my_zgemm(const char* transa, const char* transb,
              const long long* m, const long long* n, const long long* k,
              const void* alpha, const void* a, const long long* lda,
              const void* b, const long long* ldb, const void* beta,
              void* c, const long long* ldc);
#else
// LP64 interface: 32-bit integers (default)
void my_dgemm(const char* transa, const char* transb,
              const int* m, const int* n, const int* k,
              const double* alpha, const double* a, const int* lda,
              const double* b, const int* ldb, const double* beta,
              double* c, const int* ldc);

void my_zgemm(const char* transa, const char* transb,
              const int* m, const int* n, const int* k,
              const void* alpha, const void* a, const int* lda,
              const void* b, const int* ldb, const void* beta,
              void* c, const int* ldc);
#endif

// Type-specific CBLAS GEMM implementations
template <typename U, typename V, typename ResultType>
void _gemm_blas_impl(const U *A, const V *B, ResultType *C, int64_t M, int64_t N,
                     int64_t K);

// Type-specific CBLAS GEMM implementations for transpose case
template <typename U, typename V, typename ResultType>
void _gemm_blas_impl_transpose(const U *A, const V *B, ResultType *C, int64_t M,
                               int64_t N, int64_t K);

// Type-specific CBLAS GEMM implementations for conjugate case (A is conjugated)
template <typename U, typename V, typename ResultType>
void _gemm_blas_impl_conj(const U *A, const V *B, ResultType *C, int64_t M, int64_t N,
                          int64_t K);

// Specialization for double * double -> double
template <>
inline void _gemm_blas_impl<double, double, double>(const double *A,
                                                    const double *B, double *C,
                                                    int64_t M, int64_t N, int64_t K)
{
    const char transa = 'N', transb = 'N';
    const double alpha = 1.0, beta = 0.0;
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(M), ldb = static_cast<long long>(K), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(M), ldb = static_cast<int>(K), ldc = static_cast<int>(M);
#endif
    my_dgemm(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

// Specialization for complex<double> * complex<double> -> complex<double>
template <>
inline void _gemm_blas_impl<std::complex<double>, std::complex<double>,
                            std::complex<double>>(const std::complex<double> *A,
                                                  const std::complex<double> *B,
                                                  std::complex<double> *C,
                                                  int64_t M, int64_t N, int64_t K)
{
    const char transa = 'N', transb = 'N';
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(M), ldb = static_cast<long long>(K), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(M), ldb = static_cast<int>(K), ldc = static_cast<int>(M);
#endif
    my_zgemm(&transa, &transb, &m, &n, &k, reinterpret_cast<const void*>(&alpha),
               reinterpret_cast<const void*>(A), &lda,
               reinterpret_cast<const void*>(B), &ldb, reinterpret_cast<const void*>(&beta),
               reinterpret_cast<void*>(C), &ldc);
}

// Specialization for double * complex<double> -> complex<double>
template <>
inline void _gemm_blas_impl<double, std::complex<double>, std::complex<double>>(
    const double *A, const std::complex<double> *B, std::complex<double> *C,
    int64_t M, int64_t N, int64_t K)
{
    // Create temporary complex matrix from real matrix A
    std::vector<std::complex<double>> A_complex(M * K);
    for (int64_t i = 0; i < M * K; ++i) {
        A_complex[i] = std::complex<double>(A[i], 0.0);
    }

    const char transa = 'N', transb = 'N';
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(M), ldb = static_cast<long long>(K), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(M), ldb = static_cast<int>(K), ldc = static_cast<int>(M);
#endif
    my_zgemm(&transa, &transb, &m, &n, &k, reinterpret_cast<const void*>(&alpha),
               reinterpret_cast<const void*>(A_complex.data()), &lda,
               reinterpret_cast<const void*>(B), &ldb, reinterpret_cast<const void*>(&beta),
               reinterpret_cast<void*>(C), &ldc);
}

// Specialization for complex<double> * double -> complex<double>
template <>
inline void _gemm_blas_impl<std::complex<double>, double, std::complex<double>>(
    const std::complex<double> *A, const double *B, std::complex<double> *C,
    int64_t M, int64_t N, int64_t K)
{
    // Create temporary complex matrix from real matrix B
    std::vector<std::complex<double>> B_complex(K * N);
    for (int64_t i = 0; i < K * N; ++i) {
        B_complex[i] = std::complex<double>(B[i], 0.0);
    }

    const char transa = 'N', transb = 'N';
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(M), ldb = static_cast<long long>(K), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(M), ldb = static_cast<int>(K), ldc = static_cast<int>(M);
#endif
    my_zgemm(&transa, &transb, &m, &n, &k, reinterpret_cast<const void*>(&alpha),
               reinterpret_cast<const void*>(A), &lda,
               reinterpret_cast<const void*>(B_complex.data()), &ldb, reinterpret_cast<const void*>(&beta),
               reinterpret_cast<void*>(C), &ldc);
}

// Specialization for double * double -> double (transpose B)
template <>
inline void _gemm_blas_impl_transpose<double, double, double>(const double *A,
                                                              const double *B,
                                                              double *C, int64_t M,
                                                              int64_t N, int64_t K)
{
    const char transa = 'N', transb = 'T';
    const double alpha = 1.0, beta = 0.0;
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(M), ldb = static_cast<long long>(N), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(M), ldb = static_cast<int>(N), ldc = static_cast<int>(M);
#endif
    my_dgemm(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

// Specialization for complex<double> * complex<double> -> complex<double>
// (transpose B)
template <>
inline void
_gemm_blas_impl_transpose<std::complex<double>, std::complex<double>,
                          std::complex<double>>(const std::complex<double> *A,
                                                const std::complex<double> *B,
                                                std::complex<double> *C, int64_t M,
                                                int64_t N, int64_t K)
{
    const char transa = 'N', transb = 'T';
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(M), ldb = static_cast<long long>(N), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(M), ldb = static_cast<int>(N), ldc = static_cast<int>(M);
#endif
    my_zgemm(&transa, &transb, &m, &n, &k, reinterpret_cast<const void*>(&alpha),
               reinterpret_cast<const void*>(A), &lda,
               reinterpret_cast<const void*>(B), &ldb, reinterpret_cast<const void*>(&beta),
               reinterpret_cast<void*>(C), &ldc);
}

// Specialization for double * complex<double> -> complex<double> (transpose B)
template <>
inline void
_gemm_blas_impl_transpose<double, std::complex<double>, std::complex<double>>(
    const double *A, const std::complex<double> *B, std::complex<double> *C,
    int64_t M, int64_t N, int64_t K)
{
    // Use Eigen's internal gemm - it's fast enough and much simpler
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
        A_map(A, M, K);
    Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                   Eigen::Dynamic>>
        B_map(B, N, K);
    Eigen::Map<
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>
        C_map(C, M, N);

    C_map.noalias() = A_map * B_map.transpose();
}

// Specialization for complex<double> * double -> complex<double> (transpose B)
template <>
inline void
_gemm_blas_impl_transpose<std::complex<double>, double, std::complex<double>>(
    const std::complex<double> *A, const double *B, std::complex<double> *C,
    int64_t M, int64_t N, int64_t K)
{
    // Use Eigen's internal gemm - it's fast enough and much simpler
    Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                   Eigen::Dynamic>>
        A_map(A, M, K);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
        B_map(B, N, K);
    Eigen::Map<
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>
        C_map(C, M, N);

    C_map.noalias() = A_map * B_map.transpose();
}

// Specialization for double * double -> double (conjugate A, but double is real
// so same as transpose)
template <>
inline void
_gemm_blas_impl_conj<double, double, double>(const double *A, const double *B,
                                             double *C, int64_t M, int64_t N, int64_t K)
{
    const char transa = 'T', transb = 'N';
    const double alpha = 1.0, beta = 0.0;
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(K), ldb = static_cast<long long>(K), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(K), ldb = static_cast<int>(K), ldc = static_cast<int>(M);
#endif
    my_dgemm(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

// Specialization for complex<double> * complex<double> -> complex<double>
// (conjugate A)
template <>
inline void _gemm_blas_impl_conj<std::complex<double>, std::complex<double>,
                                 std::complex<double>>(
    const std::complex<double> *A, const std::complex<double> *B,
    std::complex<double> *C, int64_t M, int64_t N, int64_t K)
{
    const char transa = 'C', transb = 'N';
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(K), ldb = static_cast<long long>(K), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(K), ldb = static_cast<int>(K), ldc = static_cast<int>(M);
#endif
    my_zgemm(&transa, &transb, &m, &n, &k, reinterpret_cast<const void*>(&alpha),
               reinterpret_cast<const void*>(A), &lda,
               reinterpret_cast<const void*>(B), &ldb, reinterpret_cast<const void*>(&beta),
               reinterpret_cast<void*>(C), &ldc);
}

// Specialization for complex<double> * double -> complex<double> (conjugate A)
template <>
inline void
_gemm_blas_impl_conj<std::complex<double>, double, std::complex<double>>(
    const std::complex<double> *A, const double *B, std::complex<double> *C,
    int64_t M, int64_t N, int64_t K)
{
    // Create temporary complex matrix from real matrix B
    std::vector<std::complex<double>> B_complex(K * N);
    for (int64_t i = 0; i < K * N; ++i) {
        B_complex[i] = std::complex<double>(B[i], 0.0);
    }

    const char transa = 'C', transb = 'N';
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(K), ldb = static_cast<long long>(K), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(K), ldb = static_cast<int>(K), ldc = static_cast<int>(M);
#endif
    my_zgemm(&transa, &transb, &m, &n, &k, reinterpret_cast<const void*>(&alpha),
               reinterpret_cast<const void*>(A), &lda,
               reinterpret_cast<const void*>(B_complex.data()), &ldb, reinterpret_cast<const void*>(&beta),
               reinterpret_cast<void*>(C), &ldc);
}

// Specialization for double * complex<double> -> complex<double> (conjugate A,
// but double is real)
template <>
inline void
_gemm_blas_impl_conj<double, std::complex<double>, std::complex<double>>(
    const double *A, const std::complex<double> *B, std::complex<double> *C,
    int64_t M, int64_t N, int64_t K)
{
    // Create temporary complex matrix from real matrix A
    std::vector<std::complex<double>> A_complex(K * M);
    for (int64_t i = 0; i < K * M; ++i) {
        A_complex[i] = std::complex<double>(A[i], 0.0);
    }

    const char transa = 'T', transb = 'N';
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
#ifdef SPARSEIR_USE_BLAS_ILP64
    const long long m = static_cast<long long>(M), n = static_cast<long long>(N), k = static_cast<long long>(K);
    const long long lda = static_cast<long long>(K), ldb = static_cast<long long>(K), ldc = static_cast<long long>(M);
#else
    const int m = static_cast<int>(M), n = static_cast<int>(N), k = static_cast<int>(K);
    const int lda = static_cast<int>(K), ldb = static_cast<int>(K), ldc = static_cast<int>(M);
#endif
    my_zgemm(&transa, &transb, &m, &n, &k, reinterpret_cast<const void*>(&alpha),
               reinterpret_cast<const void*>(A_complex.data()), &lda,
               reinterpret_cast<const void*>(B), &ldb, reinterpret_cast<const void*>(&beta),
               reinterpret_cast<void*>(C), &ldc);
}

template <typename T1, typename T2, typename T3>
void _gemm_inplace(const T1 *A, const T2 *B, T3 *C, int64_t M, int64_t N, int64_t K)
{
    // use Eigen + Eigen::Map
    Eigen::Map<const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>> A_map(
        A, M, K);
    Eigen::Map<const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic>> B_map(
        B, K, N);
    Eigen::Map<Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic>> C_map(C, M,
                                                                        N);
    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl<T1, T2, T3>(A, B, C, M, N, K);
}

// second matrix is transposed
template <typename T1, typename T2, typename T3>
void _gemm_inplace_t(const T1 *A, const T2 *B, T3 *C, int64_t M, int64_t N, int64_t K)
{
    Eigen::Map<const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>> A_map(
        A, M, K);
    Eigen::Map<const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic>> B_map(
        B, N, K);
    Eigen::Map<Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic>> C_map(C, M,
                                                                        N);
    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl_transpose<T1, T2, T3>(A, B, C, M, N, K);
}

// first matrix is complex conjugated
template <typename T1, typename T2, typename T3>
void _gemm_inplace_conj(const T1 *A, const T2 *B, T3 *C, int64_t M, int64_t N, int64_t K)
{
    Eigen::Map<const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>> A_map(
        A, K, M);
    Eigen::Map<const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic>> B_map(
        B, K, N);
    Eigen::Map<Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic>> C_map(C, M,
                                                                        N);
    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl_conj<T1, T2, T3>(A, B, C, M, N, K);
}

template <typename U, typename V>
Eigen::Matrix<decltype(std::declval<U>() * std::declval<V>()), Eigen::Dynamic,
              Eigen::Dynamic>
_gemm(const Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> &A,
      const Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> &B)
{
    using ResultType = decltype(std::declval<U>() * std::declval<V>());
    const int64_t M = A.rows();
    const int64_t N = B.cols();
    const int64_t K = A.cols();

    Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> result(M, N);

    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl<U, V, ResultType>(A.data(), B.data(), result.data(), M, N,
                                      K);

    return result;
}

template <typename U, typename V>
Eigen::Matrix<decltype(std::declval<U>() * std::declval<V>()), Eigen::Dynamic,
              Eigen::Dynamic>
_gemm(
    const Eigen::Map<const Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic>> &A,
    const Eigen::Map<const Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic>> &B)
{
    using ResultType = decltype(std::declval<U>() * std::declval<V>());
    const int64_t M = A.rows();
    const int64_t N = B.cols();
    const int64_t K = A.cols();

    Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> result(M, N);

    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl<U, V, ResultType>(A.data(), B.data(), result.data(), M, N,
                                      K);

    return result;
}


} // namespace sparseir