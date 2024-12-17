// augment_test.cpp
// Ported from SparseIR.jl/test/augment.jl

#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <sparseir/sparseir-header-only.hpp>

using namespace sparseir;

// Base class for all augmentations
class AbstractAugmentation {
public:
    virtual ~AbstractAugmentation() = default;
    virtual double evaluate(double tau) const = 0;
    virtual std::complex<double> evaluate_hat(int n) const = 0;
    virtual std::shared_ptr<AbstractAugmentation> derivative(int order) const = 0;
    virtual double beta() const = 0;
};

// TauConst Augmentation
class TauConst : public AbstractAugmentation {
    double beta_;
public:
    explicit TauConst(double beta) : beta_(beta) {
        if (beta <= 0) throw std::domain_error("Temperature (beta) must be positive.");
    }

    double evaluate(double tau) const override {
        if (tau < 0 || tau > beta_)
            throw std::domain_error("Tau must be within [0, beta].");
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

// TauLinear Augmentation
class TauLinear : public AbstractAugmentation {
    double beta_, norm_;
public:
    explicit TauLinear(double beta) : beta_(beta), norm_(std::sqrt(3.0 / beta)) {
        if (beta <= 0) throw std::domain_error("Temperature (beta) must be positive.");
    }

    double evaluate(double tau) const override {
        if (tau < 0 || tau > beta_)
            throw std::domain_error("Tau must be within [0, beta].");
        double x = 2.0 * tau / beta_ - 1.0;
        return norm_ * x;
    }

    std::complex<double> evaluate_hat(int n) const override {
        if (n == 0) return 0.0;
        double inv_w = 1.0 / (M_PI / beta_ * n);
        return std::complex<double>(0, norm_ * 2.0 * inv_w);
    }

    std::shared_ptr<AbstractAugmentation> derivative(int order) const override {
        if (order == 0) return std::make_shared<TauLinear>(*this);
        if (order == 1) return nullptr; // Constant derivative
        return nullptr;
    }

    double beta() const override { return beta_; }
};

// MatsubaraConst Augmentation
class MatsubaraConst : public AbstractAugmentation {
    double beta_;
public:
    explicit MatsubaraConst(double beta) : beta_(beta) {
        if (beta <= 0) throw std::domain_error("Temperature (beta) must be positive.");
    }

    double evaluate(double tau) const override {
        if (tau < 0 || tau > beta_)
            throw std::domain_error("Tau must be within [0, beta].");
        return NAN;
    }

    std::complex<double> evaluate_hat(int /* n */) const override {
        return 1.0;
    }

    std::shared_ptr<AbstractAugmentation> derivative(int order) const override {
        return std::make_shared<MatsubaraConst>(*this);
    }

    double beta() const override { return beta_; }
};

// Augmented Basis
class AugmentedBasis {
    double beta_;
    size_t size_;
    std::vector<std::shared_ptr<AbstractAugmentation>> augmentations_;

public:
    AugmentedBasis(double beta, size_t size, std::initializer_list<std::shared_ptr<AbstractAugmentation>> augmentations)
        : beta_(beta), size_(size), augmentations_(augmentations) {}

    size_t size() const { return size_ + augmentations_.size(); }

    Eigen::VectorXd evaluate(double tau) const {
        Eigen::VectorXd result(size());
        for (size_t i = 0; i < augmentations_.size(); ++i) {
            result(i) = augmentations_[i]->evaluate(tau);
        }
        for (size_t i = augmentations_.size(); i < size(); ++i) {
            result(i) = std::sin((i + 1) * M_PI * tau / beta_); // Mock basis function
        }
        return result;
    }

    double beta() const { return beta_; }
};
}