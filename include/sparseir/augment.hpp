#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sparseir {

// AbstractAugmentation class
template <typename T>
class AbstractAugmentation {
public:
    virtual ~AbstractAugmentation() = default;

    virtual T operator()(T tau) const = 0;
    virtual T deriv(T tau, int n = 1) const = 0;
    virtual std::complex<T> hat(int n) const = 0;
    virtual std::unique_ptr<AbstractAugmentation<T>> clone() const = 0;
};

// TauConst augmentation
template <typename T>
class TauConst : public AbstractAugmentation<T> {
public:
    TauConst(T beta) : beta_(beta), norm_(1.0 / std::sqrt(beta)) { }

    T operator()(T tau) const override { return norm_; }

    T deriv(T tau, int n = 1) const override { return T(0); }

    std::complex<T> hat(int n) const override
    {
        return std::sqrt(beta_) * (n == 0 ? 1.0 : 0.0);
    }

    std::unique_ptr<AbstractAugmentation<T>> clone() const override
    {
        return std::make_unique<TauConst<T>>(*this);
    }

private:
    T beta_;
    T norm_;
};

// TauLinear augmentation
template <typename T>
class TauLinear : public AbstractAugmentation<T> {
public:
    TauLinear(T beta) : beta_(beta), norm_(std::sqrt(3 / beta)) { }

    T operator()(T tau) const override
    {
        T x = 2 / beta_ * tau - 1;
        return norm_ * x;
    }

    T deriv(T tau, int n = 1) const override
    {
        if (n == 0) {
            return (*this)(tau);
        } else if (n == 1) {
            return norm_ * 2 / beta_;
        } else {
            return T(0);
        }
    }

    std::complex<T> hat(int n) const override
    {
        std::complex<T> inv_w = M_PI / beta_ * n;
        if (n != 0) {
            inv_w = 1.0 / inv_w;
        }
        return norm_ * 2.0 / std::complex<T>(0, 1) * inv_w;
    }

    std::unique_ptr<AbstractAugmentation<T>> clone() const override
    {
        return std::make_unique<TauLinear<T>>(*this);
    }

private:
    T beta_;
    T norm_;
};

// MatsubaraConst augmentation
template <typename T>
class MatsubaraConst : public AbstractAugmentation<T> {
public:
    MatsubaraConst(T beta) : beta_(beta) { }

    T operator()(T tau) const override
    {
        return std::numeric_limits<T>::quiet_NaN(); // Undefined in tau
    }

    T deriv(T tau, int n = 1) const override
    {
        return std::numeric_limits<T>::quiet_NaN(); // Undefined in tau
    }

    std::complex<T> hat(int n) const override
    {
        return std::complex<T>(1.0, 0.0);
    }

    std::unique_ptr<AbstractAugmentation<T>> clone() const override
    {
        return std::make_unique<MatsubaraConst<T>>(*this);
    }

private:
    T beta_;
};

// _AugmentedFunction class
template <typename T>
class _AugmentedFunction {
public:
    _AugmentedFunction(
        const Eigen::VectorXd &fbasis,
        const std::vector<
            std::function<Eigen::VectorXd(const Eigen::VectorXd &)>> &faug)
        : fbasis_(fbasis), faug_(faug), naug_(faug.size())
    {
        if (fbasis_.size() == 0) {
            throw std::invalid_argument(
                "must have vector of functions as fbasis");
        }
    }

    size_t size() const { return naug_ + fbasis_.size(); }

    Eigen::MatrixXd operator()(const Eigen::VectorXd &x) const
    {
        Eigen::MatrixXd fbasis_x = fbasis_(x);
        std::vector<Eigen::MatrixXd> faug_x;
        for (const auto &faug_l : faug_) {
            faug_x.push_back(faug_l(x));
        }
        Eigen::MatrixXd f_x(fbasis_x.rows() + faug_x.size(), x.size());
        for (size_t i = 0; i < faug_x.size(); ++i) {
            f_x.row(i) = faug_x[i];
        }
        f_x.bottomRows(fbasis_x.rows()) = fbasis_x;
        return f_x;
    }

    const Eigen::VectorXd &operator[](size_t l) const
    {
        if (l < naug_) {
            return faug_[l];
        } else {
            return fbasis_[l - naug_];
        }
    }

private:
    Eigen::VectorXd fbasis_;
    std::vector<std::function<Eigen::VectorXd(const Eigen::VectorXd &)>> faug_;
    size_t naug_;
};

} // namespace sparseir