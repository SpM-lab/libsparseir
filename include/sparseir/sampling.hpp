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
        sampling_points_ = basis_->default_tau_sampling_points();
        matrix_ = eval_matrix(this, basis_, sampling_points_);
        matrix_svd_ = Eigen::JacobiSVD<Eigen::MatrixXd>(matrix_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    }

    Eigen::VectorXd fit(
        const Eigen::VectorXd& ax,
        const Eigen::VectorXd* points = nullptr) const override {
        return matrix_svd_.solve(ax);
    }

    Eigen::VectorXd evaluate(
        const Eigen::VectorXd& al,
        const Eigen::VectorXd* points = nullptr) const override {
        if (points) {
            return eval_matrix(this, basis_, *points) * al;
        }
        return matrix_ * al;
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