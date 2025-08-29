#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>
#include <complex>
#include <tuple>

#ifdef SPARSEIR_USE_BLAS
extern "C" {

    // double-precision real
    void cblas_dgemm(
        const int Order, const int TransA, const int TransB,
        const int M, const int N, const int K,
        const double alpha, const double *A, const int lda,
        const double *B, const int ldb,
        const double beta, double *C, const int ldc
    );
    
    // double-precision complex
    void cblas_zgemm(
        const int Order, const int TransA, const int TransB,
        const int M, const int N, const int K,
        const void *alpha, const void *A, const int lda,
        const void *B, const int ldb,
        const void *beta, void *C, const int ldc
    );

    enum CBLAS_TRANSPOSE {
        CblasNoTrans   = 111,
        CblasTrans     = 112,
        CblasConjTrans = 113
    };
    
    enum CBLAS_ORDER {
        CblasRowMajor = 101,
        CblasColMajor = 102
    };
}

#endif

namespace sparseir {

#ifdef SPARSEIR_USE_BLAS
// Type-specific CBLAS GEMM implementations
template<typename U, typename V, typename ResultType>
void _gemm_blas_impl(const U* A, const V* B, ResultType* C, int M, int N, int K, int lda, int ldb, int ldc);

// Specialization for double * double -> double
template<>
inline void _gemm_blas_impl<double, double, double>(const double* A, const double* B, double* C, int M, int N, int K, int lda, int ldb, int ldc) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
}

// Specialization for complex<double> * complex<double> -> complex<double>
template<>
inline void _gemm_blas_impl<std::complex<double>, std::complex<double>, std::complex<double>>(const std::complex<double>* A, const std::complex<double>* B, std::complex<double>* C, int M, int N, int K, int lda, int ldb, int ldc) {
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, 
                 reinterpret_cast<const double*>(A), lda, 
                 reinterpret_cast<const double*>(B), ldb, 
                 &beta, reinterpret_cast<double*>(C), ldc);
}

// Specialization for double * complex<double> -> complex<double>
template<>
inline void _gemm_blas_impl<double, std::complex<double>, std::complex<double>>(const double* A, const std::complex<double>* B, std::complex<double>* C, int M, int N, int K, int lda, int ldb, int ldc) {
    // Cast double to complex<double> and use zgemm
    const std::complex<double> alpha(1.0);
    const std::complex<double> beta(0.0);
    
    // Create temporary complex arrays for A
    std::vector<std::complex<double>> A_complex(M * lda);
    for (int i = 0; i < M * lda; ++i) {
        A_complex[i] = std::complex<double>(A[i], 0.0);
    }
    
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, 
                 reinterpret_cast<const double*>(A_complex.data()), lda, 
                 reinterpret_cast<const double*>(B), ldb, 
                 &beta, reinterpret_cast<double*>(C), ldc);
}

#endif

template <typename U, typename V>
Eigen::Matrix<decltype(std::declval<U>() * std::declval<V>()), Eigen::Dynamic, Eigen::Dynamic>
_gemm(
    const Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> &A,
    const Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> &B)
{
    using ResultType = decltype(std::declval<U>() * std::declval<V>());
    const int M = A.rows();
    const int N = B.cols();
    const int K = A.cols();
    
    Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> result(M, N);
    
#ifdef SPARSEIR_USE_BLAS
    // Call appropriate CBLAS function based on input types
    _gemm_blas_impl<U, V, ResultType>(A.data(), B.data(), result.data(), M, N, K, A.rows(), B.rows(), result.rows());
#else
    result = A * B;
#endif
    
    return result;
}


template <typename U, typename V>
Eigen::Matrix<decltype(std::declval<U>() * std::declval<V>()), Eigen::Dynamic, Eigen::Dynamic>
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
    _gemm_blas_impl<U, V, ResultType>(A.data(), B.data(), result.data(), M, N, K, A.rows(), B.rows(), result.rows());
#else
    result = A * B;
#endif
    
    return result;
}




template <typename T1, typename T2, int N1, int N2>
Eigen::Tensor<decltype(T1() * T2()), (N1 + N2 - 2)>
_contract(const Eigen::TensorMap<const Eigen::Tensor<T1, N1>> &tensor1,
          const Eigen::TensorMap<const Eigen::Tensor<T2, N2>> &tensor2,
          const Eigen::array<Eigen::IndexPair<int>, 1> &contract_dims)
{
    using ResultType = decltype(T1() * T2());

    // Check if the number of contract dimensions is 1
    if (contract_dims.size() != 1) {
        throw std::runtime_error("Number of contract dimensions must be 1");
    }

    // Get the contract dimensions
    int dim1 = contract_dims[0].first;
    int dim2 = contract_dims[0].second;
    
    // Move contract dimensions to the last position for tensor1 and first position for tensor2
    // Convert to Eigen::Tensor first, then apply movedim
    Eigen::Tensor<ResultType, N1> tensor1_tensor = tensor1.template cast<ResultType>();
    Eigen::Tensor<ResultType, N2> tensor2_tensor = tensor2.template cast<ResultType>();
    
    auto tensor1_moved = movedim(tensor1_tensor, dim1, N1 - 1);
    auto tensor2_moved = movedim(tensor2_tensor, dim2, 0);
    
    // Reshape to matrices for matrix multiplication
    Eigen::Index rows1 = 1, cols1 = 1, rows2 = 1, cols2 = 1;
    
    // Calculate dimensions for tensor1 (contract dim moved to last)
    for (int i = 0; i < N1 - 1; ++i) {
        rows1 *= tensor1_moved.dimension(i);
    }
    cols1 = tensor1_moved.dimension(N1 - 1);
    
    // Calculate dimensions for tensor2 (contract dim moved to first)
    cols2 = tensor2_moved.dimension(0);
    for (int i = 1; i < N2; ++i) {
        rows2 *= tensor2_moved.dimension(i);
    }
    
    // Create MatrixMaps to avoid memory copying
    Eigen::Map<const Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic>> matrix1(
        tensor1_moved.data(), rows1, cols1);
    Eigen::Map<const Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic>> matrix2(
        tensor2_moved.data(), cols2, rows2);
    
    // Perform matrix multiplication
    //Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> result_matrix = matrix1 * matrix2;
    Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> result_matrix = _gemm(matrix1, matrix2);
    
    // Convert matrix result to tensor and reshape
    Eigen::array<Eigen::Index, N1 + N2 - 2> result_dims;
    int idx = 0;
    
    // Add dimensions from tensor1 (excluding contract dim)
    for (int i = 0; i < N1 - 1; ++i) {
        result_dims[idx++] = tensor1_moved.dimension(i);
    }
    
    // Add dimensions from tensor2 (excluding contract dim)
    for (int i = 1; i < N2; ++i) {
        result_dims[idx++] = tensor2_moved.dimension(i);
    }
    
    // Convert matrix to tensor and reshape
    Eigen::Tensor<ResultType, N1 + N2 - 2> result_tensor;
    result_tensor.resize(result_dims);
    
    // Copy data from result matrix to result tensor
    std::copy(result_matrix.data(), result_matrix.data() + result_matrix.size(), result_tensor.data());
    
    return result_tensor;
}

// Overload for regular Tensor in both arguments - delegates to TensorMap
// version
template <typename T1, typename T2, int N1, int N2>
Eigen::Tensor<decltype(T1() * T2()), (N1 + N2 - 2)>
_contract(const Eigen::Tensor<T1, N1> &tensor1,
          const Eigen::Tensor<T2, N2> &tensor2,
          const Eigen::array<Eigen::IndexPair<int>, 1> &contract_dims)
{
    // Create TensorMaps from the Tensors
    Eigen::TensorMap<Eigen::Tensor<T1, N1>> tensor1_map(
        const_cast<T1 *>(tensor1.data()), tensor1.dimensions());
    Eigen::TensorMap<Eigen::Tensor<T2, N2>> tensor2_map(
        const_cast<T2 *>(tensor2.data()), tensor2.dimensions());

    // Delegate to the TensorMap version
    return _contract(tensor1_map, tensor2_map, contract_dims);
}

// Overload for TensorMap in first argument, regular Tensor in second
template <typename T1, typename T2, int N1, int N2>
Eigen::Tensor<decltype(T1() * T2()), (N1 + N2 - 2)>
_contract(const Eigen::TensorMap<const Eigen::Tensor<T1, N1>> &tensor1,
          const Eigen::Tensor<T2, N2> &tensor2,
          const Eigen::array<Eigen::IndexPair<int>, 1> &contract_dims)
{
    // Create TensorMap from the second Tensor
    Eigen::TensorMap<const Eigen::Tensor<T2, N2>> tensor2_map(
        const_cast<T2 *>(tensor2.data()), tensor2.dimensions());

    // Delegate to the TensorMap version
    return _contract(tensor1, tensor2_map, contract_dims);
}

// Overload for regular Tensor in first argument, TensorMap in second
template <typename T1, typename T2, int N1, int N2>
Eigen::Tensor<decltype(T1() * T2()), (N1 + N2 - 2)>
_contract(const Eigen::Tensor<T1, N1> &tensor1,
          const Eigen::TensorMap<const Eigen::Tensor<T2, N2>> &tensor2,
          const Eigen::array<Eigen::IndexPair<int>, 1> &contract_dims)
{
    // Create TensorMap from the first Tensor
    Eigen::TensorMap<const Eigen::Tensor<T1, N1>> tensor1_map(
        const_cast<T1 *>(tensor1.data()), tensor1.dimensions());

    // Delegate to the TensorMap version
    return _contract(tensor1_map, tensor2, contract_dims);
}

template <typename T1, typename T2, int N2>
Eigen::Tensor<decltype(T1() * T2()), N2> _matop_along_dim(
    const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const Eigen::Tensor<T2, N2> &tensor2, int dim = 0)
{
    if (dim < 0 || dim >= N2) {
        throw std::runtime_error(
            "evaluate: dimension must be in [0..N2). Got dim=" +
            std::to_string(dim));
    }

    if (matrix.cols() != tensor2.dimension(dim)) {
        throw std::runtime_error(
            "Mismatch: matrix.cols()=" + std::to_string(matrix.cols()) +
            ", but tensor2.dimension(" + std::to_string(dim) +
            ")=" + std::to_string(tensor2.dimension(dim)));
    }

    // Create a TensorMap from the matrix to avoid memory copying
    Eigen::TensorMap<const Eigen::Tensor<T1, 2>> matrix_tensor(
        matrix.data(), matrix.rows(), matrix.cols());

    // specify contraction dimensions
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, dim)};

    auto result = _contract(matrix_tensor, tensor2, contract_dims);
    return movedim(result, 0, dim);
}

// Add overload for TensorMap
template <typename T1, typename T2, int N2>
Eigen::Tensor<decltype(T1() * T2()), N2> _matop_along_dim(
    const Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const Eigen::TensorMap<const Eigen::Tensor<T2, N2>> &tensor2, int dim = 0)
{
    if (dim < 0 || dim >= N2) {
        throw std::runtime_error(
            "evaluate: dimension must be in [0..N2). Got dim=" +
            std::to_string(dim));
    }

    if (matrix.cols() != tensor2.dimension(dim)) {
        throw std::runtime_error(
            "Mismatch: matrix.cols()=" + std::to_string(matrix.cols()) +
            ", but tensor2.dimension(" + std::to_string(dim) +
            ")=" + std::to_string(tensor2.dimension(dim)));
    }

    // Create a TensorMap from the matrix to avoid memory copying
    Eigen::TensorMap<const Eigen::Tensor<T1, 2>> matrix_tensor(
        matrix.data(), matrix.rows(), matrix.cols());

    // specify contraction dimensions
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, dim)};

    auto result = _contract(matrix_tensor, tensor2, contract_dims);
    return movedim(result, 0, dim);
}

template <typename T, typename S, int N>
Eigen::Matrix<decltype(T() * S()), Eigen::Dynamic, Eigen::Dynamic>
_fit_impl_first_dim(const Eigen::JacobiSVD<Eigen::MatrixX<S>> &svd,
                    const Eigen::MatrixX<T> &B)
{
    using ResultType = decltype(T() * S());

    Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> UHB =
        svd.matrixU().adjoint() * B;

    // Apply inverse singular values to the rows of UHB
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        UHB.row(i) /= ResultType(svd.singularValues()(i));
    }
    return _gemm(svd.matrixV(), UHB);
}

inline Eigen::MatrixXcd
_fit_impl_first_dim_split_svd(const Eigen::JacobiSVD<Eigen::MatrixXcd> &svd,
                              const Eigen::MatrixXcd &B, bool has_zero)
{
    auto U = svd.matrixU();

    Eigen::Index U_halfsize =
        U.rows() % 2 == 0 ? U.rows() / 2 : U.rows() / 2 + 1;

    Eigen::MatrixXcd U_realT;
    U_realT = U.block(0, 0, U_halfsize, U.cols()).transpose();

    // Create a properly sized matrix first
    Eigen::MatrixXcd U_imag = Eigen::MatrixXcd::Zero(U_halfsize, U.cols());

    // Get the blocks we need
    if (has_zero) {
        U_imag = Eigen::MatrixXcd::Zero(U_halfsize, U.cols());
        auto U_imag_ = U.block(U_halfsize, 0, U_halfsize - 1, U.cols());
        auto U_imag_1 = U.block(0, 0, 1, U.cols());

        // Now do the assignments
        U_imag.topRows(1) = U_imag_1;
        U_imag.bottomRows(U_imag_.rows()) = U_imag_;
    } else {
        U_imag = U.block(U_halfsize, 0, U_halfsize, U.cols());
    }

    Eigen::MatrixXcd U_imagT = U_imag.transpose();
    Eigen::MatrixXcd B_real = B.real();
    Eigen::MatrixXcd B_imag = B.imag();

    Eigen::MatrixXcd UHB = _gemm(U_realT, B_real);
    UHB += _gemm(U_imagT, B_imag);

    // Apply inverse singular values to the rows of UHB
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        UHB.row(i) /= std::complex<double>(svd.singularValues()(i));
    }
    return _gemm(svd.matrixV(), UHB);
}

template <typename T, typename S, int N>
Eigen::Tensor<decltype(T() * S()), N>
fit_impl(const Eigen::JacobiSVD<Eigen::MatrixX<S>> &svd,
         const Eigen::Tensor<T, N> &arr, int dim)
{
    if (dim < 0 || dim >= N) {
        throw std::domain_error("Dimension must be in [0, N).");
    }

    // First move the dimension to the first
    auto arr_ = movedim(arr, dim, 0);
    // Create a view of the tensor as a matrix
    Eigen::MatrixX<T> arr_view = Eigen::Map<Eigen::MatrixX<T>>(
        arr_.data(), arr_.dimension(0), arr_.size() / arr_.dimension(0));
    // output matrix size
    Eigen::MatrixX<T> result = _fit_impl_first_dim<T, S, N>(svd, arr_view);
    // Copy the result to a tensor
    Eigen::array<Eigen::Index, N> dims;
    dims[0] = result.rows();
    for (int i = 1; i < N; ++i) {
        dims[i] = arr_.dimension(i);
    }
    Eigen::Tensor<T, N> result_tensor(dims);
    std::copy(result.data(), result.data() + result.size(),
              result_tensor.data());

    return movedim(result_tensor, 0, dim);
}

template <int N>
Eigen::Tensor<std::complex<double>, N>
fit_impl_split_svd(const Eigen::JacobiSVD<Eigen::MatrixXcd> &svd,
                   const Eigen::Tensor<std::complex<double>, N> &arr, int dim,
                   bool has_zero)
{
    if (dim < 0 || dim >= N) {
        throw std::domain_error("Dimension must be in [0, N).");
    }

    // First move the dimension to the first
    auto arr_ = movedim(arr, dim, 0);
    // Create a view of the tensor as a matrix
    Eigen::MatrixXcd arr_view = Eigen::Map<Eigen::MatrixXcd>(
        arr_.data(), arr_.dimension(0), arr_.size() / arr_.dimension(0));
    Eigen::MatrixXcd result =
        _fit_impl_first_dim_split_svd(svd, arr_view, has_zero);
    // Copy the result to a tensor
    Eigen::array<Eigen::Index, N> dims;
    dims[0] = result.rows();
    for (int i = 1; i < N; ++i) {
        dims[i] = arr_.dimension(i);
    }
    Eigen::Tensor<std::complex<double>, N> result_tensor(dims);
    std::copy(result.data(), result.data() + result.size(),
              result_tensor.data());

    return movedim(result_tensor, 0, dim);
}

}