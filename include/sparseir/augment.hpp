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
        std::invalid_argument("TauConst is not a Fermionic basis.");
        return std::numeric_limits<std::complex<double>>::quiet_NaN();
    }

    std::function<double(double)> deriv(int order = 1) const override {
        if (order == 0) {
            return [this](double tau) { return (*this)(tau); };
        }
        return [](double) { return 0.0; };
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
        std::invalid_argument("TauConst is not a Fermionic basis.");
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::function<double(double)> deriv(int order = 1) const override {
        if (order == 0) {
            return [this](double tau) { return (*this)(tau); };
        } else if (order == 1) {
            return [this](double) { return norm * 2.0 / beta; };
        }
        return [](double) { return 0.0; };
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

// AugmentedBasis
template <typename S, typename B, typename F, typename FHAT>
class AugmentedBasis : public AbstractBasis<S> {
public:
    std::shared_ptr<FiniteTempBasis<B>> basis;
    std::vector<std::shared_ptr<AbstractAugmentation>> augmentations;
    F u;
    FHAT uhat;

    AugmentedBasis(std::shared_ptr<FiniteTempBasis<B>> basis,
                   std::vector<std::shared_ptr<AbstractAugmentation>> augmentations,
                   F u, FHAT uhat)
        : basis(basis), augmentations(augmentations), u(u), uhat(uhat) {}

    size_t size() const {
        return nAug() + basis->s.size();
    }

    size_t nAug() const {
        return augmentations.size();
    }

    double accuracy() const override {
        // TODO: Implement accuracy calculation
        return 0;
        //return basis->accuracy();
    }

    double omegaMax() const {
        return basis->omegaMax();
    }

    const Eigen::VectorXd significance() const override{
        // TODO: Implement significance calculation
        return Eigen::VectorXd::Zero(10);
    };

    static std::shared_ptr<AugmentedBasis> create(std::shared_ptr<B> basis,
                                                  std::vector<std::shared_ptr<AbstractAugmentation>> augs) {
        auto u = createAugmentedTauFunction(basis->u, augs);
        auto uhat = createAugmentedMatsubaraFunction(basis->uhat, augs);
        return std::make_shared<AugmentedBasis>(basis, augs, u, uhat);
    }

private:
    static F createAugmentedTauFunction(const F &basisFunc, const std::vector<std::shared_ptr<AbstractAugmentation>> &augmentations) {
        // Placeholder for actual implementation
        return basisFunc;
    }

    static FHAT createAugmentedMatsubaraFunction(const FHAT &basisFunc, const std::vector<std::shared_ptr<AbstractAugmentation>> &augmentations) {
        // Placeholder for actual implementation
        return basisFunc;
    }
};


} // namespace sparseir
