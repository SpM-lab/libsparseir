#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>
#include <complex>
#include <tuple>

#include "sparseir/jacobi_svd.hpp"

#ifdef SPARSEIR_USE_BLAS
extern "C" {

// double-precision real
void cblas_dgemm(const int Order, const int TransA, const int TransB,
                 const int M, const int N, const int K, const double alpha,
                 const double *A, const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);

// double-precision complex
void cblas_zgemm(const int Order, const int TransA, const int TransB,
                 const int M, const int N, const int K, const void *alpha,
                 const void *A, const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc);

enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
}

#endif

namespace sparseir {

#ifdef SPARSEIR_USE_BLAS
// Type-specific CBLAS GEMM implementations
template <typename U, typename V, typename ResultType>
void _gemm_blas_impl(const U *A, const V *B, ResultType *C, int M, int N,
                     int K);

// Type-specific CBLAS GEMM implementations for transpose case
template <typename U, typename V, typename ResultType>
void _gemm_blas_impl_transpose(const U *A, const V *B, ResultType *C, int M,
                               int N, int K);

// Type-specific CBLAS GEMM implementations for conjugate case (A is conjugated)
template <typename U, typename V, typename ResultType>
void _gemm_blas_impl_conj(const U *A, const V *B, ResultType *C, int M, int N,
                          int K);

// Specialization for double * double -> double
template <>
inline void _gemm_blas_impl<double, double, double>(const double *A,
                                                    const double *B, double *C,
                                                    int M, int N, int K)
{
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, M,
                B, K, 0.0, C, M);
}

// Specialization for complex<double> * complex<double> -> complex<double>
template <>
inline void _gemm_blas_impl<std::complex<double>, std::complex<double>,
                            std::complex<double>>(const std::complex<double> *A,
                                                  const std::complex<double> *B,
                                                  std::complex<double> *C,
                                                  int M, int N, int K)
{
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha,
                reinterpret_cast<const double *>(A), M,
                reinterpret_cast<const double *>(B), K, &beta,
                reinterpret_cast<double *>(C), M);
}

// Specialization for double * complex<double> -> complex<double>
template <>
inline void _gemm_blas_impl<double, std::complex<double>, std::complex<double>>(
    const double *A, const std::complex<double> *B, std::complex<double> *C,
    int M, int N, int K)
{
    // Create temporary complex matrix from real matrix A
    std::vector<std::complex<double>> A_complex(M * K);
    for (int i = 0; i < M * K; ++i) {
        A_complex[i] = std::complex<double>(A[i], 0.0);
    }

    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha,
                reinterpret_cast<const double *>(A_complex.data()), M,
                reinterpret_cast<const double *>(B), K, &beta,
                reinterpret_cast<double *>(C), M);
}

// Specialization for complex<double> * double -> complex<double>
template <>
inline void _gemm_blas_impl<std::complex<double>, double, std::complex<double>>(
    const std::complex<double> *A, const double *B, std::complex<double> *C,
    int M, int N, int K)
{
    // Create temporary complex matrix from real matrix B
    std::vector<std::complex<double>> B_complex(K * N);
    for (int i = 0; i < K * N; ++i) {
        B_complex[i] = std::complex<double>(B[i], 0.0);
    }

    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha,
                reinterpret_cast<const double *>(A), M,
                reinterpret_cast<const double *>(B_complex.data()), K, &beta,
                reinterpret_cast<double *>(C), M);
}

// Specialization for double * double -> double (transpose B)
template <>
inline void _gemm_blas_impl_transpose<double, double, double>(const double *A,
                                                              const double *B,
                                                              double *C, int M,
                                                              int N, int K)
{
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, A, M, B,
                N, 0.0, C, M);
}

// Specialization for complex<double> * complex<double> -> complex<double>
// (transpose B)
template <>
inline void
_gemm_blas_impl_transpose<std::complex<double>, std::complex<double>,
                          std::complex<double>>(const std::complex<double> *A,
                                                const std::complex<double> *B,
                                                std::complex<double> *C, int M,
                                                int N, int K)
{
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, &alpha,
                reinterpret_cast<const double *>(A), M,
                reinterpret_cast<const double *>(B), N, &beta,
                reinterpret_cast<double *>(C), M);
}

// Specialization for double * complex<double> -> complex<double> (transpose B)
template <>
inline void
_gemm_blas_impl_transpose<double, std::complex<double>, std::complex<double>>(
    const double *A, const std::complex<double> *B, std::complex<double> *C,
    int M, int N, int K)
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
    int M, int N, int K)
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
                                             double *C, int M, int N, int K)
{
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0, A, K, B,
                K, 0.0, C, M);
}

// Specialization for complex<double> * complex<double> -> complex<double>
// (conjugate A)
template <>
inline void _gemm_blas_impl_conj<std::complex<double>, std::complex<double>,
                                 std::complex<double>>(
    const std::complex<double> *A, const std::complex<double> *B,
    std::complex<double> *C, int M, int N, int K)
{
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, M, N, K, &alpha,
                reinterpret_cast<const double *>(A), K,
                reinterpret_cast<const double *>(B), K, &beta,
                reinterpret_cast<double *>(C), M);
}

// Specialization for complex<double> * double -> complex<double> (conjugate A)
template <>
inline void
_gemm_blas_impl_conj<std::complex<double>, double, std::complex<double>>(
    const std::complex<double> *A, const double *B, std::complex<double> *C,
    int M, int N, int K)
{
    // Create temporary complex matrix from real matrix B
    std::vector<std::complex<double>> B_complex(K * N);
    for (int i = 0; i < K * N; ++i) {
        B_complex[i] = std::complex<double>(B[i], 0.0);
    }

    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, M, N, K, &alpha,
                reinterpret_cast<const double *>(A), K,
                reinterpret_cast<const double *>(B_complex.data()), K, &beta,
                reinterpret_cast<double *>(C), M);
}

// Specialization for double * complex<double> -> complex<double> (conjugate A,
// but double is real)
template <>
inline void
_gemm_blas_impl_conj<double, std::complex<double>, std::complex<double>>(
    const double *A, const std::complex<double> *B, std::complex<double> *C,
    int M, int N, int K)
{
    // Create temporary complex matrix from real matrix A
    std::vector<std::complex<double>> A_complex(K * M);
    for (int i = 0; i < K * M; ++i) {
        A_complex[i] = std::complex<double>(A[i], 0.0);
    }

    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
    cblas_zgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, &alpha,
                reinterpret_cast<const double *>(A_complex.data()), K,
                reinterpret_cast<const double *>(B), K, &beta,
                reinterpret_cast<double *>(C), M);
}

#endif

template <typename T1, typename T2, typename T3>
void _gemm_inplace(const T1 *A, const T2 *B, T3 *C, int M, int N, int K)
{
    // use Eigen + Eigen::Map
    Eigen::Map<const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>> A_map(
        A, M, K);
    Eigen::Map<const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic>> B_map(
        B, K, N);
    Eigen::Map<Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic>> C_map(C, M,
                                                                        N);
#ifdef SPARSEIR_USE_BLAS
    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl<T1, T2, T3>(A, B, C, M, N, K);
#else
    C_map.noalias() = A_map * B_map;
#endif
}

// second matrix is transposed
template <typename T1, typename T2, typename T3>
void _gemm_inplace_t(const T1 *A, const T2 *B, T3 *C, int M, int N, int K)
{
    Eigen::Map<const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>> A_map(
        A, M, K);
    Eigen::Map<const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic>> B_map(
        B, N, K);
    Eigen::Map<Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic>> C_map(C, M,
                                                                        N);
#ifdef SPARSEIR_USE_BLAS
    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl_transpose<T1, T2, T3>(A, B, C, M, N, K);
#else
    C_map.noalias() = A_map * B_map.transpose();
#endif
}

// first matrix is complex conjugated
template <typename T1, typename T2, typename T3>
void _gemm_inplace_conj(const T1 *A, const T2 *B, T3 *C, int M, int N, int K)
{
    Eigen::Map<const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic>> A_map(
        A, K, M);
    Eigen::Map<const Eigen::Matrix<T2, Eigen::Dynamic, Eigen::Dynamic>> B_map(
        B, K, N);
    Eigen::Map<Eigen::Matrix<T3, Eigen::Dynamic, Eigen::Dynamic>> C_map(C, M,
                                                                        N);
#ifdef SPARSEIR_USE_BLAS
    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl_conj<T1, T2, T3>(A, B, C, M, N, K);
#else
    C_map.noalias() = A_map.adjoint() * B_map;
#endif
}

template <typename U, typename V>
Eigen::Matrix<decltype(std::declval<U>() * std::declval<V>()), Eigen::Dynamic,
              Eigen::Dynamic>
_gemm(const Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> &A,
      const Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> &B)
{
    using ResultType = decltype(std::declval<U>() * std::declval<V>());
    const int M = A.rows();
    const int N = B.cols();
    const int K = A.cols();

    Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> result(M, N);

#ifdef SPARSEIR_USE_BLAS
    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl<U, V, ResultType>(A.data(), B.data(), result.data(), M, N,
                                      K);
#else
    result.noalias() = A * B;
#endif

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
    const int M = A.rows();
    const int N = B.cols();
    const int K = A.cols();

    Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> result(M, N);

#ifdef SPARSEIR_USE_BLAS
    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl<U, V, ResultType>(A.data(), B.data(), result.data(), M, N,
                                      K);
#else
    result.noalias() = A * B;
#endif

    return result;
}


} // namespace sparseir