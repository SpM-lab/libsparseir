// Template class for FiniteTempBasis
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <memory> // for std::shared_ptr
#include <vector>

namespace sparseir {

template <typename S>
class AbstractAugmentation {
public:
    virtual ~AbstractAugmentation() = default;

    /**
     * @brief Evaluate the function at imaginary time τ.
     *
     * @param tau Imaginary time τ in [0, β].
     * @return The value of the function at τ.
     */
    virtual double operator()(double tau) const = 0;

    /**
     * @brief Evaluate the function at a Matsubara frequency.
     *
     * @param n Matsubara frequency index.
     * @return The value of the function at Matsubara frequency n.
     */
    virtual std::complex<double> operator()(const MatsubaraFreq<S>& n) const = 0;

    /**
     * @brief Get the inverse temperature β.
     *
     * @return The inverse temperature β.
     */
    virtual double beta() const = 0;

    /**
     * @brief Compute the derivative of the function.
     *
     * @param order Order of the derivative.
     * @return A function representing the derivative.
     */
    virtual std::function<double(double)> derivative(int order = 1) const = 0;
};

/**
 * @brief Augmented basis combining the given basis with augmentation functions.
 *
 * The augmented functions form the first basis functions, followed by the regular basis functions.
 *
 * @tparam S Statistical type (Fermionic or Bosonic).
 */
template <typename S>
class AugmentedBasis : public AbstractBasis<S> {
public:
    using BasisPtr = std::shared_ptr<AbstractBasis<S>>;
    using AugmentationPtr = std::shared_ptr<AbstractAugmentation<S>>;

    /**
     * @brief Constructor for AugmentedBasis.
     *
     * @param basis Shared pointer to the original basis.
     * @param augmentations Vector of shared pointers to augmentation functions.
     */
    AugmentedBasis(BasisPtr basis, const std::vector<AugmentationPtr>& augmentations)
        : basis_(basis), augmentations_(augmentations) {
        if (!basis_) {
            throw std::invalid_argument("Basis pointer cannot be null.");
        }
    }

    /**
     * @brief Get the number of augmented functions.
     *
     * @return The number of augmentations.
     */
    size_t naug() const {
        return augmentations_.size();
    }

    /**
     * @brief Get the total size of the augmented basis.
     *
     * @return The total number of basis functions.
     */
    size_t size() const override {
        return naug() + basis_->size();
    }

    /**
     * @brief Evaluate the l-th basis function at imaginary time τ.
     *
     * @param l Index of the basis function (0-based).
     * @param tau Imaginary time τ in [0, β].
     * @return The value of the l-th basis function at τ.
     */
    double u(int l, double tau) const override {
        if (l < static_cast<int>(naug())) {
            return (*augmentations_[l])(tau);
        } else {
            return basis_->u(l - naug(), tau);
        }
    }

    /**
     * @brief Evaluate the l-th basis function at Matsubara frequency n.
     *
     * @param l Index of the basis function (0-based).
     * @param n Matsubara frequency index.
     * @return The value of the l-th basis function at Matsubara frequency n.
     */
    std::complex<double> uhat(int l, const MatsubaraFreq<S>& n) const override {
        if (l < static_cast<int>(naug())) {
            return (*augmentations_[l])(n);
        } else {
            return basis_->uhat(l - naug(), n);
        }
    }

    /**
     * @brief Get the statistical type (Fermionic or Bosonic).
     *
     * @return The statistical type.
     */
    S statistics() const override {
        return basis_->statistics();
    }

    /**
     * @brief Get the inverse temperature β.
     *
     * @return The inverse temperature β.
     */
    double beta() const override {
        return basis_->beta();
    }

    /**
     * @brief Get the maximum frequency ωmax.
     *
     * @return The maximum frequency ωmax.
     */
    double omega_max() const override {
        return basis_->omega_max();
    }

    /**
     * @brief Get the significances vector.
     *
     * @return Significances of the augmented basis.
     */
    const Eigen::VectorXd significance() {
        Eigen::VectorXd sig_aug(naug());
        sig_aug.setOnes();  // Augmentation functions have significance 1
        Eigen::VectorXd sig = basis_->significance();
        Eigen::VectorXd combined_sig(sig_aug.size() + sig.size());
        combined_sig << sig_aug, sig;
        return combined_sig;
    }

    /**
     * @brief Returns true if the sampling is expected to be well-conditioned.
     *
     * @return True if well-conditioned.
     */
    bool is_well_conditioned() const {
        // Implementation specific; assuming same as basis
        return basis_->is_well_conditioned();
    }

private:
    BasisPtr basis_;
    std::vector<AugmentationPtr> augmentations_;
};

/**
 * @brief Constant function in imaginary time, defined in [0, β].
 */
template <typename S>
class TauConst : public AbstractAugmentation<S> {
public:
    /**
     * @brief Constructor for TauConst.
     *
     * @param beta Inverse temperature β.
     */
    explicit TauConst(double beta) : beta_(beta) {
        if (beta_ <= 0.0) {
            throw std::domain_error("Temperature must be positive.");
        }
        norm_ = 1.0 / std::sqrt(beta_);
    }

    double operator()(double tau) const override {
        if (tau < 0.0 || tau > beta_) {
            throw std::domain_error("τ must be in [0, β].");
        }
        return norm_;
    }

    std::complex<double> operator()(const MatsubaraFreq<S>& n) const override {
        if (n.isZero()) {
            return std::sqrt(beta_);
        } else {
            return 0.0;
        }
    }

    double beta() const override {
        return beta_;
    }

    std::function<double(double)> derivative(int order = 1) const override {
        if (order == 0) {
            return [this](double tau) -> double {
                return this->operator()(tau);
            };
        } else {
            return [](double tau) -> double {
                return 0.0;
            };
        }
    }

private:
    double beta_;
    double norm_;
};

/**
 * @brief Linear function in imaginary time, antisymmetric around β/2.
 */
template <typename S>
class TauLinear : public AbstractAugmentation<S> {
public:
    /**
     * @brief Constructor for TauLinear.
     *
     * @param beta Inverse temperature β.
     */
    explicit TauLinear(double beta) : beta_(beta) {
        if (beta_ <= 0.0) {
            throw std::domain_error("Temperature must be positive.");
        }
        norm_ = std::sqrt(3.0 / beta_);
    }

    double operator()(double tau) const override {
        if (tau < 0.0 || tau > beta_) {
            throw std::domain_error("τ must be in [0, β].");
        }
        double x = 2.0 / beta_ * tau - 1.0;
        return norm_ * x;
    }

    std::complex<double> operator()(const MatsubaraFreq<S>& n) const override {
        if (n.isZero()) {
            return 0.0;
        }
        std::complex<double> iw = n.value(beta_);
        return norm_ * 2.0 / (std::complex<double>(0.0, 1.0) * iw);
    }

    double beta() const override {
        return beta_;
    }

    std::function<double(double)> derivative(int order = 1) const override {
        if (order == 0) {
            return [this](double tau) -> double {
                return this->operator()(tau);
            };
        } else if (order == 1) {
            return [this](double tau) -> double {
                return norm_ * 2.0 / beta_;
            };
        } else {
            return [](double tau) -> double {
                return 0.0;
            };
        }
    }

private:
    double beta_;
    double norm_;
};

/**
 * @brief Constant function in Matsubara frequencies, undefined in imaginary time.
 */
template <typename S>
class MatsubaraConst : public AbstractAugmentation<S> {
public:
    /**
     * @brief Constructor for MatsubaraConst.
     *
     * @param beta Inverse temperature β.
     */
    explicit MatsubaraConst(double beta) : beta_(beta) {
        if (beta_ <= 0.0) {
            throw std::domain_error("Temperature must be positive.");
        }
    }

    double operator()(double tau) const override {
        if (tau < 0.0 || tau > beta_) {
            throw std::domain_error("τ must be in [0, β].");
        }
        // Undefined in imaginary time; return NaN
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::complex<double> operator()(const MatsubaraFreq<S>& n) const override {
        return 1.0;
    }

    double beta() const override {
        return beta_;
    }

    std::function<double(double)> derivative(int order = 1) const override {
        // Derivative remains the same for MatsubaraConst
        return [this](double tau) -> double {
            return this->operator()(tau);
        };
    }

private:
    double beta_;
};

} // namespace sparseir
