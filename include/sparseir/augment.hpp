#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sparseir {

// Abstract Augmentation
class AbstractAugmentation : public std::enable_shared_from_this<AbstractAugmentation> {
public:
    virtual double operator()(double tau) const = 0;
    virtual std::complex<double> operator()(MatsubaraFreq<Bosonic> n) const = 0;
    virtual std::complex<double> operator()(MatsubaraFreq<Fermionic> n) const = 0;
    virtual std::function<double(double)> deriv(int order = 1) const = 0;
    virtual std::shared_ptr<AbstractAugmentation> create(
        const std::shared_ptr<AbstractBasis<Bosonic>>& basis) const = 0;
    virtual ~AbstractAugmentation() = default;
};

// TauConst Class
class TauConst : public AbstractAugmentation {
public:
    double beta;

    explicit TauConst(double beta) : beta(beta) {
        if (beta <= 0) {
            throw std::domain_error("Temperature must be positive.");
        }
    }

    TauConst(const AbstractBasis<Bosonic>& basis) : TauConst(basis.beta) {}


    double operator()(double tau) const override {
        if (tau < 0 || tau > beta) {
            throw std::domain_error("tau must be in [0, beta].");
        }
        return 1.0 / std::sqrt(beta);
    }

    std::complex<double> operator()(MatsubaraFreq<Bosonic> n) const override {
        return is_zero(n) ? std::sqrt(beta) : 0.0;
    }

    std::complex<double> operator()(MatsubaraFreq<Fermionic> n) const override {
        throw std::invalid_argument("TauConst is not a Fermionic basis.");
        return std::numeric_limits<std::complex<double>>::quiet_NaN();
    }

    std::function<double(double)> deriv(int order = 1) const override {
        if (order == 0) {
            return [this](double tau) { return (*this)(tau); };
        }
        return [](double) { return 0.0; };
    }

    std::shared_ptr<AbstractAugmentation> create(
        const std::shared_ptr<AbstractBasis<Bosonic>>& basis) const override {
        return std::make_shared<TauConst>(basis->beta);
    }
};

// TauLinear Class
class TauLinear : public AbstractAugmentation {
public:
    double beta;
    double norm;

    explicit TauLinear(double beta) : beta(beta), norm(std::sqrt(3.0 / beta)) {
        if (beta <= 0) {
            throw std::domain_error("Temperature must be positive.");
        }
    }

    TauLinear(const AbstractBasis<Bosonic>& basis) : TauLinear(basis.beta) {}

    TauLinear(const std::shared_ptr<AbstractBasis<Bosonic>>& basis)
        : TauLinear(basis->beta) {}

    double operator()(double tau) const override {
        if (tau < 0 || tau > beta) {
            throw std::domain_error("tau must be in [0, beta].");
        }
        double x = 2.0 / beta * tau - 1.0;
        return norm * x;
    }

    std::complex<double> operator()(MatsubaraFreq<Bosonic> n) const override {
        double inv_w = n.n * (M_PI / beta);
        inv_w = is_zero(n) ? inv_w : 1.0 / inv_w;
        return norm * 2.0 / std::complex<double>(0, 1) * inv_w;
    }

    std::complex<double> operator()(MatsubaraFreq<Fermionic> n) const override {
        throw std::invalid_argument("TauConst is not a Fermionic basis.");
        return std::numeric_limits<std::complex<double>>::quiet_NaN();
    }

    std::function<double(double)> deriv(int order = 1) const override {
        if (order == 0) {
            return [this](double tau) { return (*this)(tau); };
        } else if (order == 1) {
            return [this](double) { return norm * 2.0 / beta; };
        }
        return [](double) { return 0.0; };
    }

    std::shared_ptr<AbstractAugmentation> create(
        const std::shared_ptr<AbstractBasis<Bosonic>>& basis) const override {
        return std::make_shared<TauLinear>(basis->beta);
    }
};

// MatsubaraConst Class
class MatsubaraConst : public AbstractAugmentation {
public:
    double beta;

    explicit MatsubaraConst(double beta) : beta(beta) {
        if (beta <= 0) {
            throw std::domain_error("Temperature must be positive.");
        }
    }

    MatsubaraConst(const AbstractBasis<Bosonic>& basis) : MatsubaraConst(basis.beta) {}


    double operator()(double tau) const override {
        if (tau < 0 || tau > beta) {
            throw std::domain_error("tau must be in [0, beta].");
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::complex<double> operator()(MatsubaraFreq<Bosonic> n) const override {
        return 1.0;
    }

    std::complex<double> operator()(MatsubaraFreq<Fermionic> n) const override {
        return 1.0;
    }

    std::function<double(double)> deriv(int order = 1) const override {
        return [this](double tau) { return (*this)(tau); };
    }

    std::shared_ptr<AbstractAugmentation> create(
        const std::shared_ptr<AbstractBasis<Bosonic>>& basis) const override {
        return std::make_shared<MatsubaraConst>(basis->beta);
    }
};

// AbstractAugmentedFunction
class AbstractAugmentedFunction {
public:
    virtual size_t size() const = 0;
    virtual Eigen::VectorXd operator()(double x) const = 0;
    virtual Eigen::MatrixXd operator()(const Eigen::VectorXd &x) const = 0;
    virtual ~AbstractAugmentedFunction() = default;
};

// AugmentedFunction
template <typename FB, typename FA>
class AugmentedFunction : public AbstractAugmentedFunction {
public:
    FB fbasis;
    std::vector<FA> faug;

    AugmentedFunction(FB fbasis, std::vector<FA> faug) : fbasis(fbasis), faug(faug) {}

    size_t size() const override {
        return faug.size() + fbasis.size();
    }

    Eigen::VectorXd operator()(double x) const override {
        Eigen::VectorXd fbasis_x = fbasis(x);
        Eigen::VectorXd faug_x(faug.size());
        for (size_t i = 0; i < faug.size(); ++i) {
            faug_x[i] = faug[i](x);
        }
        Eigen::VectorXd result(faug.size() + fbasis_x.size());
        result << faug_x, fbasis_x;
        return result;
    }

    Eigen::MatrixXd operator()(const Eigen::VectorXd &x) const override {
        Eigen::MatrixXd fbasis_x = fbasis(x);
        Eigen::MatrixXd faug_x(faug.size(), x.size());
        for (size_t i = 0; i < faug.size(); ++i) {
            for (size_t j = 0; j < x.size(); ++j) {
                faug_x(i, j) = faug[i](x[j]);
            }
        }
        Eigen::MatrixXd result(faug.size() + fbasis_x.rows(), x.size());
        result << faug_x, fbasis_x;
        return result;
    }
};

// Add before AugmentedBasis class definition

class AugmentedTauFunction {
public:
    const PiecewiseLegendrePolyVector& basis_func;
    const std::vector<std::shared_ptr<AbstractAugmentation>>& augmentations;

    AugmentedTauFunction(const PiecewiseLegendrePolyVector& basis_func,
                        const std::vector<std::shared_ptr<AbstractAugmentation>>& augmentations)
        : basis_func(basis_func)
        , augmentations(augmentations) {}

    Eigen::VectorXd operator()(double tau) const {
        Eigen::VectorXd result = basis_func(tau);
        for (const auto& aug : augmentations) {
            result.conservativeResize(result.size() + 1);
            result(result.size() - 1) = (*aug)(tau);
        }
        return result;
    }

    size_t size() const {
        return augmentations.size() + basis_func.size();
    }
};

template <typename S>
class AugmentedMatsubaraFunction {
public:
    using MatsubaraVec = PiecewiseLegendreFTVector<S>;

    const MatsubaraVec& basis_func;
    const std::vector<std::shared_ptr<AbstractAugmentation>>& augmentations;

    AugmentedMatsubaraFunction(const MatsubaraVec& basis_func,
                              const std::vector<std::shared_ptr<AbstractAugmentation>>& augmentations)
        : basis_func(basis_func)
        , augmentations(augmentations) {}

    template<typename T>
    std::complex<double> operator()(MatsubaraFreq<T> n) const {
        std::complex<double> result = basis_func(n);
        for (const auto& aug : augmentations) {
            result += (*aug)(n);
        }
        return result;
    }

    size_t size() const {
        return augmentations.size() + basis_func.size();
    }
};

// AugmentedBasis
template <typename S>
class AugmentedBasis : public AbstractBasis<S> {
private:
    std::shared_ptr<FiniteTempBasis<S>> basis_;
    std::vector<std::shared_ptr<AbstractAugmentation>> augmentations_;
public:
    std::unique_ptr<AugmentedTauFunction> u;
    std::unique_ptr<AugmentedMatsubaraFunction<S>> uhat;

    AugmentedBasis(std::shared_ptr<FiniteTempBasis<S>> basis,
                  const std::vector<std::shared_ptr<AbstractAugmentation>>& augmentations)
        : basis_(basis)
        , augmentations_(augmentations)
        , u(std::make_unique<AugmentedTauFunction>(basis->u, augmentations))
        , uhat(std::make_unique<AugmentedMatsubaraFunction<S>>(basis->uhat, augmentations)) {}

    // Prevent copying, allow moving
    AugmentedBasis(const AugmentedBasis&) = delete;
    AugmentedBasis& operator=(const AugmentedBasis&) = delete;
    AugmentedBasis(AugmentedBasis&&) = default;
    AugmentedBasis& operator=(AugmentedBasis&&) = default;

    // Implement pure virtual functions
    size_t size() const override {
        return augmentations_.size() + basis_->size();
    }

    double get_accuracy() const override {
        return basis_->get_accuracy();
    }

    double get_wmax() const override {
        return basis_->get_wmax();
    }

    const Eigen::VectorXd significance() const override {
        return basis_->significance();
    }

    size_t nAug() const { return augmentations_.size(); }

    const Eigen::VectorXd default_tau_sampling_points() const override {
        int sz = this->basis_->sve_result->s.size() + this->augmentations_.size();
        auto x = default_sampling_points(this->basis_->sve_result->u, sz);
        return (this->basis_->get_beta() / 2.0) * (x.array() + 1.0);
    }

    // Factory method
    static std::shared_ptr<AugmentedBasis<S>> create(
        std::shared_ptr<FiniteTempBasis<S>> basis,
        const std::vector<std::shared_ptr<AbstractAugmentation>>& augmentations) {
        return std::make_shared<AugmentedBasis<S>>(basis, augmentations);
    }
};

} // namespace sparseir
