#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace sparseir {

// Custom SVD class that can be constructed from precomputed results
// This avoids the need to recompute SVD when we already have the results
template<typename MatrixType>
class JacobiSVD {
private:
    MatrixType U_;
    MatrixType V_;
    Eigen::VectorXd singular_values_;

public:
    // Constructor from precomputed results
    JacobiSVD(const MatrixType& U, const Eigen::VectorXd& singular_values, const MatrixType& V)
        : U_(U), V_(V), singular_values_(singular_values) {}
    
    // Default constructor (for compatibility)
    JacobiSVD() = default;
    
    // Constructor that computes SVD (for compatibility with Eigen::JacobiSVD)
    JacobiSVD(const MatrixType& matrix, unsigned int computationOptions) {
        Eigen::JacobiSVD<MatrixType> eigen_svd(matrix, computationOptions);
        U_ = eigen_svd.matrixU();
        V_ = eigen_svd.matrixV();
        singular_values_ = eigen_svd.singularValues();
    }
    
    // Interface compatible with Eigen::JacobiSVD
    const MatrixType& matrixU() const { return U_; }
    const MatrixType& matrixV() const { return V_; }
    const Eigen::VectorXd& singularValues() const { return singular_values_; }
    
    // For compatibility with existing code that might call compute()
    void compute(const MatrixType& matrix, unsigned int computationOptions) {
        Eigen::JacobiSVD<MatrixType> eigen_svd(matrix, computationOptions);
        U_ = eigen_svd.matrixU();
        V_ = eigen_svd.matrixV();
        singular_values_ = eigen_svd.singularValues();
    }
};

} // namespace sparseir
