#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sparseir {

template <typename S>
class AbstractSampling {
public:
    virtual ~AbstractSampling() = default;

    // Evaluate the basis coefficients at sampling points
    virtual Eigen::VectorXd evaluate(
        const Eigen::VectorXd& al,
        const Eigen::VectorXd* points = nullptr) const = 0;

    // Fit values at sampling points to basis coefficients
    virtual Eigen::VectorXd fit(
        const Eigen::VectorXd& ax,
        const Eigen::VectorXd* points = nullptr) const = 0;

    // Get the sampling points
    virtual const Eigen::VectorXd& sampling_points() const = 0;
};
// Helper function declarations
// Forward declarations
template <typename S>
class TauSampling;

template <typename S>
inline Eigen::MatrixXd eval_matrix(const TauSampling<S>* tau_sampling,
                           const std::shared_ptr<FiniteTempBasis<S>>& basis,
                           const Eigen::VectorXd& x){
    return basis->u(x).transpose();
}

template <typename S>
class TauSampling : public AbstractSampling<S> {
public:
    TauSampling(
        const std::shared_ptr<FiniteTempBasis<S>>& basis) : basis_(basis) {
        // Get default sampling points from basis
        sampling_points_ = basis_->default_tau_sampling_points();

        // Ensure matrix dimensions are correct
        if (sampling_points_.size() == 0) {
            throw std::runtime_error("No sampling points generated");
        }

        // Initialize evaluation matrix with correct dimensions
        matrix_ = eval_matrix(this, basis_, sampling_points_);
        std::cout << "matrix_ = " << matrix_ << std::endl;
        // Check matrix dimensions
        if (matrix_.rows() != sampling_points_.size() ||
            matrix_.cols() != basis_->size()) {
            throw std::runtime_error(
                "Matrix dimensions mismatch: got " +
                std::to_string(matrix_.rows()) + "x" +
                std::to_string(matrix_.cols()) +
                ", expected " + std::to_string(sampling_points_.size()) +
                "x" + std::to_string(basis_->size()));
        }

        // Initialize SVD
        matrix_svd_ = Eigen::JacobiSVD<Eigen::MatrixXd>(
            matrix_,
            Eigen::ComputeFullU | Eigen::ComputeFullV
        );
    }

    Eigen::VectorXd evaluate(
        const Eigen::VectorXd& al,
        const Eigen::VectorXd* points = nullptr) const override {
        if (points) {
            auto eval_mat = eval_matrix(this, basis_, *points);
            if (eval_mat.cols() != al.size()) {
                throw std::runtime_error(
                    "Input vector size mismatch: got " +
                    std::to_string(al.size()) +
                    ", expected " + std::to_string(eval_mat.cols()));
            }
            return eval_mat * al;
        }

        if (matrix_.cols() != al.size()) {
            throw std::runtime_error(
                "Input vector size mismatch: got " +
                std::to_string(al.size()) +
                ", expected " + std::to_string(matrix_.cols()));
        }
        return matrix_ * al;
    }

    Eigen::VectorXd fit(
        const Eigen::VectorXd& ax,
        const Eigen::VectorXd* points = nullptr) const override {
        if (points) {
            auto eval_mat = eval_matrix(this, basis_, *points);
            Eigen::JacobiSVD<Eigen::MatrixXd> local_svd(
                eval_mat,
                Eigen::ComputeFullU | Eigen::ComputeFullV
            );
            return local_svd.solve(ax);
        }

        if (ax.size() != matrix_.rows()) {
            throw std::runtime_error(
                "Input vector size mismatch: got " +
                std::to_string(ax.size()) +
                ", expected " + std::to_string(matrix_.rows()));
        }
        return matrix_svd_.solve(ax);
    }

    const Eigen::VectorXd& sampling_points() const override {
        return sampling_points_;
    }

    const Eigen::VectorXd& tau() const {
        return sampling_points_;
    }

private:
    std::shared_ptr<FiniteTempBasis<S>> basis_;
    Eigen::VectorXd sampling_points_;
    Eigen::MatrixXd matrix_;
    Eigen::JacobiSVD<Eigen::MatrixXd> matrix_svd_;
};


} // namespace sparseir