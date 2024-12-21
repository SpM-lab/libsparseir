#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sparseir {

template <typename T>
class AbstractAugmentation {
public:
    virtual T operator()(T tau) const = 0; // Ensure this is virtual
    virtual T deriv(T tau, int n = 1) const = 0; // Add this line
    virtual std::complex<T> hat(int n) const = 0; // Add this line
    virtual ~AbstractAugmentation() = default;
    virtual std::unique_ptr<AbstractAugmentation<T>> clone() const = 0;
};

template <typename S>
class AugmentedBasis : public AbstractBasis<S> {
public:
    AugmentedBasis(std::shared_ptr<AbstractBasis<S>> basis,
                   const std::vector<std::unique_ptr<AbstractAugmentation<S>>>& augmentations)
        : _basis(basis), _augmentations(augmentations), _naug(augmentations.size()) {
        //Error Handling: Check for null basis pointer
        if (!_basis) {
            throw std::invalid_argument("Basis cannot be null");
        }
        //Check for valid augmentations
        for (const auto& aug : _augmentations) {
            if (!aug) {
                throw std::invalid_argument("Augmentation cannot be null");
            }
        }
    }

    size_t size() const { return _basis->size() + _naug; }

    Eigen::VectorXd u(const Eigen::VectorXd& tau) const {
        Eigen::VectorXd result(size());
        for (size_t i = 0; i < _naug; ++i) {
            result(i) = (*_augmentations[i])(tau(i));
        }
        for (size_t i = _naug; i < size(); ++i) {
            result(i) = _basis->u(tau(i - _naug))(i - _naug);
        }
        return result;
    }

    Eigen::VectorXcf uhat(const Eigen::VectorXcf& wn) const {
        Eigen::VectorXcf result(size());
        for (size_t i = 0; i < _naug; ++i) {
            result(i) = (*_augmentations[i]).hat(wn(i));
        }
        for (size_t i = _naug; i < size(); ++i) {
            result(i) = _basis->uhat(wn(i - _naug))(i - _naug);
        }
        return result;
    }

    Eigen::VectorXd v(const Eigen::VectorXd& w) const {
        return _basis->v(w);
    }

    Eigen::VectorXd s() const { return _basis->s(); }

    double beta() const { return _basis->beta(); }

    double wmax() const { return _basis->wmax(); }

    std::shared_ptr<Statistics> statistics() const {
        return _basis->statistics(); // Assuming _basis also returns a shared_ptr
    }

private:
    std::shared_ptr<AbstractBasis<S>> _basis;
    std::vector<std::unique_ptr<AbstractAugmentation<S>>> _augmentations;
    size_t _naug;
};


template <typename T>
class TauConst : public AbstractAugmentation<T> {
public:
    TauConst(T beta) : beta_(beta), norm_(1.0 / std::sqrt(beta)) {
        if (beta_ <= 0) {
            throw std::invalid_argument("beta must be positive");
        }
    }

    T operator()(T tau) const override {
        check_tau_range(tau, beta_);
        return norm_;
    }
    T deriv(T tau, int n = 1) const  {
        if (n == 0) return (*this)(tau);
        return 0.0;
    }
    std::complex<T> hat(int n) const  {
        return norm_ * std::sqrt(beta_);
    }
    std::unique_ptr<AbstractAugmentation<T>> clone() const override {
        return std::make_unique<TauConst<T>>(*this);
    }

private:
    T beta_;
    T norm_;
};

template <typename T>
class TauLinear : public AbstractAugmentation<T> {
public:
    TauLinear(T beta) : beta_(beta), norm_(std::sqrt(3.0 / beta)) {
        if (beta_ <= 0) {
            throw std::invalid_argument("beta must be positive");
        }
    }

    T operator()(T tau) const override {
        check_tau_range(tau, beta_);
        return norm_ * (2.0 * tau / beta_ - 1.0);
    }
    T deriv(T tau, int n = 1) const override {
        if (n == 1) return norm_ * 2.0 / beta_;
        return 0.0;
    }
    std::complex<T> hat(int n) const override {
        if (n == 0) return 0.0;
        return norm_ * 2.0 * beta_ / (n * M_PI * std::complex<T>(0, 1));
    }
    std::unique_ptr<AbstractAugmentation<T>> clone() const override {
        return std::make_unique<TauLinear<T>>(*this);
    }

private:
    T beta_;
    T norm_;
};

template <typename T>
class MatsubaraConst : public AbstractAugmentation<T> {
public:
    MatsubaraConst(T beta) : beta_(beta) {
        if (beta_ <= 0) {
            throw std::invalid_argument("beta must be positive");
        }
    }

    T operator()(T tau) const override { return std::numeric_limits<T>::quiet_NaN(); }
    T deriv(T tau, int n = 1) const override { return std::numeric_limits<T>::quiet_NaN(); }
    std::complex<T> hat(int n) const override { return 1.0; }
    std::unique_ptr<AbstractAugmentation<T>> clone() const override {
        return std::make_unique<MatsubaraConst<T>>(*this);
    }

private:
    T beta_;
};


} // namespace sparseir
