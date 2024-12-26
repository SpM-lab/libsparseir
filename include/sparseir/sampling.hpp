#pragma once

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sparseir {

template <typename S>
class AbstractSampling {
public:
    virtual ~AbstractSampling() = default;

    // Evaluate the basis coefficients at sampling points
    virtual Eigen::Matrix<double, Eigen::Dynamic, 1> evaluate(
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& al,
        const Eigen::VectorXd* points = nullptr) const = 0;

    // Fit values at sampling points to basis coefficients
    virtual Eigen::Matrix<double, Eigen::Dynamic, 1> fit(
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& ax,
        const Eigen::VectorXd* points = nullptr) const = 0;

    // Condition number of the transformation matrix
    virtual double cond() const = 0;

    // Get the sampling points
    virtual const Eigen::VectorXd& sampling_points() const = 0;

    // Get the transformation matrix
    virtual const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& matrix() const = 0;

    // Get the associated basis
    //virtual const std::shared_ptr<AbstractBasis<S>>& basis() const = 0;

protected:
    // Create a new sampling object for different sampling points
    virtual std::shared_ptr<AbstractSampling> for_sampling_points(
        const Eigen::VectorXd& x) const = 0;
};

template <typename S>
class TauSampling : public AbstractSampling<S> {
public:
    TauSampling(
        const std::shared_ptr<AbstractBasis<S>>& basis,
        const Eigen::VectorXd* sampling_points = nullptr)
        : basis_(basis) {
        if (sampling_points) {
            sampling_points_ = *sampling_points;
        } else {
            sampling_points_ = default_tau_sampling_points(basis_);
        }
        construct_matrix();
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> evaluate(
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& al,
        const Eigen::VectorXd* points = nullptr) const override {
        if (points) {
            auto sampling = for_sampling_points(*points);
            return sampling->matrix() * al;
        }
        return matrix_ * al;
    }

    /*
    Eigen::Matrix<double, Eigen::Dynamic, 1> fit(
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& ax,
        const Eigen::VectorXd* points = nullptr) const override {
        if (points) {
            auto sampling = for_sampling_points(*points);
            return sampling->solve(ax);
        }
        return solve(ax);
    }
    */

    double cond() const override {
        return cond_;
    }

    const Eigen::VectorXd& sampling_points() const override {
        return sampling_points_;
    }

    const Eigen::VectorXd& tau() const {
        return sampling_points_;
    }

    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& matrix() const override {
        return matrix_;
    }
    /*
    const std::shared_ptr<AbstractBasis<double>>& basis() const override {
        return basis_;
    }
    */

protected:
    std::shared_ptr<AbstractSampling<S>> for_sampling_points(
        const Eigen::VectorXd& x) const override {
        return std::make_shared<TauSampling>(basis_, &x);
    }

private:
    void construct_matrix() {
        matrix_ = basis_->u(sampling_points_).transpose();
        cond_ = compute_condition_number(matrix_);
        solver_.compute(matrix_);
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("Matrix decomposition failed.");
        }
    }

    Eigen::Matrix<double, Eigen::Dynamic, 1> solve(
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& ax) const {
        return solver_.solve(ax);
    }

    double compute_condition_number(
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& mat) const {
        Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> svd(
            mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double cond = svd.singularValues()(0) /
                      svd.singularValues()(svd.singularValues().size() - 1);
        return cond;
    }

    std::shared_ptr<AbstractBasis<S>> basis_;
    Eigen::VectorXd sampling_points_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_;
    mutable Eigen::ColPivHouseholderQR<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> solver_;
    double cond_;
};

template <typename T>
class MatsubaraSampling : public AbstractSampling<std::complex<T>> {
public:
    MatsubaraSampling(
        const std::shared_ptr<AbstractBasis<T>>& basis,
        bool positive_only = false,
        const Eigen::VectorXi* sampling_points = nullptr)
        : basis_(basis), positive_only_(positive_only) {
        if (sampling_points) {
            sampling_points_ = *sampling_points;
        } else {
            sampling_points_ = basis_->default_matsubara_sampling_points(positive_only);
        }
        construct_matrix();
    }

    /*
    Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> evaluate(
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& al,
        const Eigen::VectorXi* points = nullptr) const override {
        if (points) {
            // TODO: Implement for_sampling_points
            auto sampling = for_sampling_points(*points);
            return sampling->matrix() * al;
        }
        return matrix_ * al;
    }
    */


    double cond() const override {
        return cond_;
    }

    const Eigen::VectorXi& sampling_points() const override {
        return sampling_points_;
    }

    const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>& matrix() const override {
        return matrix_;
    }

    const std::shared_ptr<AbstractBasis<T>>& basis() const override {
        return basis_;
    }

private:
    void construct_matrix() {
        matrix_ = basis_->uhat(sampling_points_);
        cond_ = compute_condition_number(matrix_);
        solver_.compute(matrix_);
        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("Matrix decomposition failed.");
        }
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> solve(
        const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>& ax) const {
        return solver_.solve(ax.real());
    }

    double compute_condition_number(
        const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>& mat) const {
        Eigen::JacobiSVD<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>> svd(
            mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double cond = svd.singularValues()(0).real() /
                      svd.singularValues()(svd.singularValues().size() - 1).real();
        return cond;
    }

    std::shared_ptr<AbstractBasis<T>> basis_;
    bool positive_only_;
    Eigen::VectorXi sampling_points_;
    Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic> matrix_;
    mutable Eigen::ColPivHouseholderQR<Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic>> solver_;
    double cond_;
};

} // namespace sparseir