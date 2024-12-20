#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace sparseir {

// Forward declaration of AbstractAugmentation
template <typename T>
class AbstractAugmentation;

// AbstractAugmentation base class
template <typename T>
class AbstractAugmentation {
public:
    virtual ~AbstractAugmentation() { }

    // Evaluate the augmentation function at point x
    virtual T operator()(T x) const = 0;

    // Evaluate the derivative of the augmentation function at point x
    // n is the order of the derivative
    virtual T deriv(T x, int n = 1) const = 0;

    // Evaluate the Fourier transform of the augmentation function at Matsubara
    // frequency n
    virtual std::complex<T> hat(int n) const = 0;

    // Clone method for creating copies of derived classes
    virtual std::unique_ptr<AbstractAugmentation<T>> clone() const = 0;
};

// Example augmentation functions
// TauConst augmentation (constant function in tau)
template <typename T>
class TauConst : public AbstractAugmentation<T> {
public:
    TauConst(T beta) : beta_(beta), norm_(1.0 / std::sqrt(beta)) { }

    virtual T operator()(T tau) const override { return norm_; }

    virtual T deriv(T tau, int n = 1) const override { return T(0); }

    virtual std::complex<T> hat(int n) const override
    {
        int zeta = 1; // Fermionic (zeta = 1) or Bosonic (zeta = 0)
        if (n == 0 && zeta == 0) {
            return norm_ * beta_; // Handle n=0 separately for bosonic case
        }
        T wn = M_PI * n / beta_;
        return norm_ * beta_ * 2.0 * wn /
               (wn * wn * beta_ * beta_ + (M_PI * M_PI));
    }

    virtual std::unique_ptr<AbstractAugmentation<T>> clone() const override
    {
        return std::make_unique<TauConst<T>>(*this);
    }

private:
    T beta_;
    T norm_;
};

// MatsubaraConst augmentation (constant in Matsubara frequency)
template <typename T>
class MatsubaraConst : public AbstractAugmentation<T> {
public:
    MatsubaraConst(T beta) : beta_(beta) { }

    virtual T operator()(T tau) const override
    {
        return std::numeric_limits<T>::quiet_NaN(); // Undefined in tau
    }

    virtual T deriv(T tau, int n = 1) const override
    {
        return std::numeric_limits<T>::quiet_NaN(); // Undefined in tau
    }

    virtual std::complex<T> hat(int n) const override
    {
        return std::complex<T>(1.0, 0.0);
    }

    virtual std::unique_ptr<AbstractAugmentation<T>> clone() const override
    {
        return std::make_unique<MatsubaraConst<T>>(*this);
    }

private:
    T beta_;
};

// AugmentedBasis class
template <typename T>
class AugmentedBasis {
public:
    using BasisPtr = std::shared_ptr<AbstractBasis<T>>;
    using AugmentationPtr = std::unique_ptr<AbstractAugmentation<T>>;

    AugmentedBasis(BasisPtr basis,
                   const std::vector<AugmentationPtr> &augmentations)
        : basis_(basis)
    {
        for (const auto &aug : augmentations) {
            augmentations.push_back(aug->clone());
        }
    }

    // Evaluate the l-th basis function at imaginary time tau
    T u(size_t l, T tau) const
    {
        if (l < augmentations.size()) {
            return (*augmentations[l])(tau);
        } else {
            return basis_->u(l - augmentations.size(), tau);
        }
    }

    // Evaluate the l-th basis function at Matsubara frequency n
    std::complex<T> uhat(size_t l, int n) const
    {
        if (l < augmentations.size()) {
            return augmentations[l]->hat(n);
        } else {
            return basis_->uhat(l - augmentations.size(), n);
        }
    }

    size_t size() const { return augmentations.size() + basis_->size(); }

private:
    BasisPtr basis_;
    std::vector<AugmentationPtr> augmentations;
};

} // namespace sparseir