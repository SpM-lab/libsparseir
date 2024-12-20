#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace sparseir {

// Abstract class representing a scalar function in imaginary time/frequency
class AbstractAugmentation {
public:
    virtual ~AbstractAugmentation() {}

    // Create an augmentation compatible with the given basis
    virtual std::unique_ptr<AbstractAugmentation> create(const AbstractBasis& basis) const = 0;

    // Get inverse temperature β
    virtual double beta() const = 0;

    // Evaluate the function at given imaginary times τ
    virtual Eigen::ArrayXd operator()(const Eigen::ArrayXd& tau) const = 0;

    // Evaluate the Matsubara transform at given frequencies n
    virtual Eigen::ArrayXcd hat(const Eigen::ArrayXi& n) const = 0;
};

/*
// Shared pointer for augmentations
using AugmentationPtr = std::shared_ptr<AbstractAugmentation>;
// List of augmentations
using AugmentationList = std::vector<AugmentationPtr>;

// Augmented basis on the imaginary-time/frequency axis
class AugmentedBasis : public AbstractBasis {
public:
    AugmentedBasis(std::shared_ptr<AbstractBasis> basis, const AugmentationList& augmentations)
        : basis_(basis), augmentations_(augmentations) {}

    // Total size of the augmented basis
    size_t size() const override {
        return augmentations_.size() + basis_->size();
    }

    // Access inverse temperature β
    double beta() const override {
        return basis_->beta();
    }

    // Access cutoff frequency ω_max
    double wmax() const override {
        return basis_->wmax();
    }

    // Access singular values (augmentations have singular value 1)
    Eigen::VectorXd singular_values() const override {
        Eigen::VectorXd sv_aug(augmentations_.size());
        sv_aug.setOnes();
        Eigen::VectorXd sv = basis_->singular_values();
        Eigen::VectorXd result(sv_aug.size() + sv.size());
        result << sv_aug, sv;
        return result;
    }

    // Evaluate basis functions at imaginary times τ
    Eigen::ArrayXXd u(const Eigen::ArrayXd& tau) const override {
        size_t n_aug = augmentations_.size();
        size_t n_basis = basis_->size();
        Eigen::ArrayXXd result(size(), tau.size());

        // Evaluate augmentations
        for (size_t i = 0; i < n_aug; ++i) {
            result.row(i) = (*augmentations_[i])(tau);
        }

        // Evaluate basis functions
        result.block(n_aug, 0, n_basis, tau.size()) = basis_->u(tau);

        return result;
    }

    // Evaluate basis functions at Matsubara frequencies n
    Eigen::ArrayXXcd uhat(const Eigen::ArrayXi& n) const override {
        size_t n_aug = augmentations_.size();
        size_t n_basis = basis_->size();
        Eigen::ArrayXXcd result(size(), n.size());

        // Evaluate augmentations
        for (size_t i = 0; i < n_aug; ++i) {
            result.row(i) = augmentations_[i]->hat(n);
        }

        // Evaluate basis functions
        result.block(n_aug, 0, n_basis, n.size()) = basis_->uhat(n);

        return result;
    }

    // Get the underlying basis
    const std::shared_ptr<AbstractBasis>& basis() const {
        return basis_;
    }

    // Get the list of augmentations
    const AugmentationList& augmentations() const {
        return augmentations_;
    }

private:
    std::shared_ptr<AbstractBasis> basis_;
    AugmentationList augmentations_;
};

// Augmentation representing a constant function in τ
class TauConst : public AbstractAugmentation {
public:
    explicit TauConst(double beta) : beta_(beta) {}

    std::unique_ptr<AbstractAugmentation> create(const AbstractBasis& basis) const override {
        return std::make_unique<TauConst>(basis.beta());
    }

    double beta() const override {
        return beta_;
    }

    Eigen::ArrayXd operator()(const Eigen::ArrayXd& tau) const override {
        return Eigen::ArrayXd::Constant(tau.size(), 1.0 / std::sqrt(beta_));
    }

    Eigen::ArrayXcd hat(const Eigen::ArrayXi& n) const override {
        // For bosonic frequencies, the Fourier transform of a constant is δ(ω_n)
        // For n = 0, δ(ω_n) = √β; otherwise, it's zero
        return (n == 0).select(std::sqrt(beta_), 0.0);
    }

private:
    double beta_;
};

// Augmentation representing a linear function in τ
class TauLinear : public AbstractAugmentation {
public:
    explicit TauLinear(double beta) : beta_(beta), norm_(std::sqrt(12.0 / std::pow(beta_, 3))) {}

    std::unique_ptr<AbstractAugmentation> create(const AbstractBasis& basis) const override {
        return std::make_unique<TauLinear>(basis.beta());
    }

    double beta() const override {
        return beta_;
    }

    Eigen::ArrayXd operator()(const Eigen::ArrayXd& tau) const override {
        return norm_ * (tau - beta_ / 2.0);
    }

    Eigen::ArrayXcd hat(const Eigen::ArrayXi& n) const override {
        Eigen::ArrayXcd iw = Eigen::ArrayXcd(0, M_PI * n.cast<double>() / beta_);
        return norm_ * beta_ * iw.inverse() * std::sqrt(-1.0) * (iw.sin() - iw * beta_ / 2.0 * iw.cos());
    }

private:
    double beta_;
    double norm_;
};

// Augmentation representing a constant function in Matsubara frequencies
class MatsubaraConst : public AbstractAugmentation {
public:
    explicit MatsubaraConst(double beta) : beta_(beta) {}

    std::unique_ptr<AbstractAugmentation> create(const AbstractBasis& basis) const override {
        return std::make_unique<MatsubaraConst>(basis.beta());
    }

    double beta() const override {
        return beta_;
    }

    Eigen::ArrayXd operator()(const Eigen::ArrayXd& tau) const override {
        // The inverse Fourier transform of the Matsubara constant
        return Eigen::ArrayXd::Constant(tau.size(), 1.0 / beta_);
    }

    Eigen::ArrayXcd hat(const Eigen::ArrayXi& n) const override {
        return Eigen::ArrayXcd::Ones(n.size());
    }

private:
    double beta_;
};
*/
} // namespace sparseir