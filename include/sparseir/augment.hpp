// augment.hpp
// Ported from SparseIR.jl/src/augment.jl

#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <Eigen/Dense>

namespace sparseir {

// Abstract base class for augmentations
class AbstractAugmentation {
public:
    virtual ~AbstractAugmentation() = default;
    virtual double beta() const = 0;
    virtual double omega_max() const = 0;
    virtual Eigen::VectorXd significance() const = 0;
    virtual bool is_well_conditioned() const = 0;
    virtual size_t size() const = 0;
    virtual double u(int l, double tau) const = 0;
};

// TauConst: Constant function in imaginary time
class TauConst : public AbstractAugmentation {
    double beta_;
public:
    explicit TauConst(double beta) : beta_(beta) {
        if (beta <= 0) throw std::domain_error("Beta must be positive.");
    }

    double evaluate(double tau) const override {
        if (tau < 0 || tau > beta_)
            throw std::domain_error("Tau must be in [0, beta].");
        return 1.0 / std::sqrt(beta_);
    }

    std::complex<double> evaluate_hat(int n) const override {
        return (n == 0) ? std::sqrt(beta_) : 0.0;
    }

    std::shared_ptr<AbstractAugmentation> derivative(int order) const override {
        if (order == 0) return std::make_shared<TauConst>(*this);
        return nullptr;
    }

    double beta() const override { return beta_; }
};

// TauLinear: Linear function in imaginary time
class TauLinear : public AbstractAugmentation {
    double beta_, norm_;
public:
    explicit TauLinear(double beta) : beta_(beta), norm_(std::sqrt(3.0 / beta)) {
        if (beta <= 0) throw std::domain_error("Beta must be positive.");
    }

    double evaluate(double tau) const override {
        if (tau < 0 || tau > beta_)
            throw std::domain_error("Tau must be in [0, beta].");
        double x = 2.0 / beta_ * tau - 1.0;
        return norm_ * x;
    }

    std::complex<double> evaluate_hat(int n) const override {
        double inv_w = (n == 0) ? 0 : 1.0 / (M_PI / beta_ * n);
        return std::complex<double>(0, norm_ * 2 * inv_w);
    }

    std::shared_ptr<AbstractAugmentation> derivative(int order) const override {
        if (order == 0) return std::make_shared<TauLinear>(*this);
        if (order == 1) return nullptr; // Constant derivative
        return nullptr;
    }

    double beta() const override { return beta_; }
};

// Augmented Basis
template <typename BasisType>
class AugmentedBasis {
    BasisType basis_;
    std::vector<std::shared_ptr<AbstractAugmentation>> augmentations_;
    size_t naug_;

public:
    explicit AugmentedBasis(BasisType basis,
                            const std::vector<std::shared_ptr<AbstractAugmentation>>& augmentations)
        : basis_(basis), augmentations_(augmentations), naug_(augmentations.size()) {}

    size_t size() const { return naug_ + basis_.size(); }

    // Evaluate basis functions
    Eigen::VectorXd evaluate(double tau) const {
        Eigen::VectorXd result(size());
        for (size_t i = 0; i < naug_; ++i)
            result[i] = augmentations_[i]->evaluate(tau);
        for (size_t i = 0; i < basis_.size(); ++i)
            result[i + naug_] = basis_.evaluate(i, tau);
        return result;
    }

    // Sampling points (simple example)
    std::vector<double> default_tau_sampling_points(int npoints) const {
        std::vector<double> points(npoints);
        double step = basis_.beta() / (npoints - 1);
        for (int i = 0; i < npoints; ++i) points[i] = i * step;
        return points;
    }

    double beta() const override { return basis_.beta(); }
    size_t augmentations_count() const override { return naug_; }
};

} // namespace sparseir