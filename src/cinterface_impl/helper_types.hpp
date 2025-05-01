#include "sparseir/sparseir.hpp"
#include <memory>

class AbstractMatsubaraFunctions {
public:
    virtual ~AbstractMatsubaraFunctions() = default;
    virtual Eigen::MatrixXcd
    operator()(const Eigen::ArrayXi &n_array) const = 0;
    virtual int size() const = 0;
};

template <typename InternalType>
class MatsubaraBasisFunctions : public AbstractMatsubaraFunctions {
private:
    std::shared_ptr<InternalType> impl;

public:
    MatsubaraBasisFunctions(std::shared_ptr<InternalType> impl) : impl(impl) { }

    virtual Eigen::MatrixXcd
    operator()(const Eigen::ArrayXi &n_array) const override
    {
        return impl->operator()(n_array);
    }

    virtual int size() const override { return impl->size(); }
};

// Abstract class for functions of a single variable (in the imaginary time
// domain or the real frequency domain)
class AbstractContinuousFunctions {
public:
    virtual ~AbstractContinuousFunctions() = default;
    virtual Eigen::VectorXd operator()(double x) const = 0;
    virtual int size() const = 0;
    virtual std::pair<double, double> get_domain() const = 0;
};

class PiecewiseLegendrePolyFunctions : public AbstractContinuousFunctions {
private:
    std::shared_ptr<sparseir::PiecewiseLegendrePolyVector> impl;

public:
    PiecewiseLegendrePolyFunctions(std::shared_ptr<sparseir::PiecewiseLegendrePolyVector> impl) : impl(impl) { }

    virtual Eigen::VectorXd operator()(double x) const override
    {
        return impl->operator()(x);
    }

    virtual int size() const override { return impl->size(); }

    virtual std::pair<double, double> get_domain() const override
    {
        return std::make_pair(impl->xmin(), impl->xmax());
    }
};

template <typename S>
int _sign()
{
    if (std::is_same<S, sparseir::Fermionic>::value) {
        return -1;
    }
    return 1;
}

// Accept a tau in [-beta, beta] and return a tau in [0, beta] with a sign
// change for the fermionic case
//
// Interpreted as:
// -beta -> -beta^+
//  beta -> beta^-
//  +0.0 -> 0^+
//  -0.0 -> 0^-
inline std::pair<double, int> regularize_tau(double tau, double beta, int fermionic_sign)
{
    // Check if tau is in the valid range
    if (tau < -beta) {
        throw std::invalid_argument("tau is less than -beta");
    }
    if (tau > beta) {
        std::cout << "tau " << tau << " is greater than beta " << beta << std::endl;
        throw std::invalid_argument("tau is greater than beta");
    }

    if (0 < tau && tau <= beta) {
        return std::make_pair(tau, 0);
    }

    if (-beta <= tau && tau < 0) {
        return std::make_pair(tau + beta, fermionic_sign);
    }

    if (tau == 0) {
        // Check if tau is -0.0
        if (std::signbit(tau)) {
            return std::make_pair(beta, fermionic_sign);
        } else {
            return std::make_pair(0.0, 0);
        }
    }

    throw std::invalid_argument("Something is wrong with the input tau");
}


template <typename ImplType>
class OmegaFunctions : public AbstractContinuousFunctions {
private:
    std::shared_ptr<ImplType> impl;

public:
    OmegaFunctions(std::shared_ptr<ImplType> impl)
        : impl(impl)
    {
    }

    virtual Eigen::VectorXd operator()(double x) const override
    {
        return impl->operator()(x);
    }

    virtual int size() const override { return impl->size(); }

    virtual std::pair<double, double> get_domain() const override
    {
        return impl->get_domain();
    }
};


template <typename ImplType>
class TauFunctions : public AbstractContinuousFunctions {
private:
    std::shared_ptr<ImplType> impl;
    double beta;
    int fermionic_sign;

public:
    TauFunctions(std::shared_ptr<ImplType> impl, double beta, int fermionic_sign)
        : impl(impl), beta(beta), fermionic_sign(fermionic_sign)
    {
    }

    virtual Eigen::VectorXd operator()(double x) const override
    {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        std::pair<double, int> regularized_tau = regularize_tau(x, beta, fermionic_sign);
        std::cout << "regularized_tau: " << regularized_tau.first << ", " << regularized_tau.second << std::endl;
        return impl->operator()(regularized_tau.first) * regularized_tau.second;
    }

    virtual int size() const override { return impl->size(); }

    virtual std::pair<double, double> get_domain() const override
    {
        return std::make_pair(-beta, beta);
    }

    int get_fermionic_sign() const { return fermionic_sign; }
};


class AbstractFiniteTempBasis {
public:
    virtual ~AbstractFiniteTempBasis() = default;
    virtual int size() const = 0;
    virtual double get_beta() const = 0;
    virtual spir_statistics_type get_statistics() const = 0;
    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const = 0;
    virtual std::shared_ptr<AbstractContinuousFunctions> get_v() const = 0;
    virtual std::shared_ptr<AbstractMatsubaraFunctions> get_uhat() const = 0;
};

template <typename S>
class _FiniteTempBasis : public AbstractFiniteTempBasis {
private:
    std::shared_ptr<sparseir::FiniteTempBasis<S>> impl;

public:
    _FiniteTempBasis(std::shared_ptr<sparseir::FiniteTempBasis<S>> impl)
        : impl(impl)
    {
    }

    virtual double get_beta() const override { return impl->get_beta(); }

    virtual int size() const override { return impl->size(); }

    virtual spir_statistics_type get_statistics() const override
    {
        if (std::is_same<S, sparseir::Fermionic>::value) {
            return SPIR_STATISTICS_FERMIONIC;
        } else {
            return SPIR_STATISTICS_BOSONIC;
        }
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const override
    {
        using ImplType = sparseir::PiecewiseLegendrePolyVector;
        int fermionic_sign = _sign<S>();
        auto u_tau_funcs = std::make_shared<TauFunctions<ImplType>>(impl->u, get_beta(), fermionic_sign);
        return std::static_pointer_cast<AbstractContinuousFunctions>(u_tau_funcs);
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_v() const override
    {
        return std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<PiecewiseLegendrePolyFunctions>(impl->v));
    }

    virtual std::shared_ptr<AbstractMatsubaraFunctions>
    get_uhat() const override
    {
        return std::static_pointer_cast<AbstractMatsubaraFunctions>(
            std::make_shared<MatsubaraBasisFunctions<
                sparseir::PiecewiseLegendreFTVector<S>>>(impl->uhat));
    }

    std::shared_ptr<sparseir::FiniteTempBasis<S>> get_impl() const
    {
        return impl;
    }
};

class AbstractDLR {
public:
    virtual ~AbstractDLR() = default;
    virtual int size() const = 0;
    virtual double get_beta() const = 0;
    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const = 0;
    virtual std::shared_ptr<AbstractMatsubaraFunctions> get_uhat() const = 0;
    virtual spir_statistics_type get_statistics() const = 0;
    virtual std::vector<double> get_poles() const = 0;
};


template <typename S>
class _DLR : public AbstractDLR {
private:
    std::shared_ptr<sparseir::DiscreteLehmannRepresentation<S>> impl;

public:
    _DLR(std::shared_ptr<sparseir::DiscreteLehmannRepresentation<S>> impl)
        : impl(impl)
    {
    }

    virtual int size() const override {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        return impl->size();
    }

    virtual spir_statistics_type get_statistics() const override
    {
        if (std::is_same<S, sparseir::Fermionic>::value) {
            return SPIR_STATISTICS_FERMIONIC;
        } else {
            return SPIR_STATISTICS_BOSONIC;
        }
    }

    virtual double get_beta() const override {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        return impl->get_beta();
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const override
    {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        using ImplType = sparseir::TauPoles<S>;
        return std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<TauFunctions<ImplType>>(impl->u, get_beta(), _sign<S>()));
    }

    virtual std::shared_ptr<AbstractMatsubaraFunctions>
    get_uhat() const override
    {
        if (!impl) {
            throw std::runtime_error("impl is not initialized");
        }
        return std::static_pointer_cast<AbstractMatsubaraFunctions>(
            std::make_shared<
                MatsubaraBasisFunctions<sparseir::MatsubaraPoles<S>>>(
                impl->uhat));
    }

    virtual std::vector<double> get_poles() const override
    {
        Eigen::VectorXd eigen_poles = impl->poles;
        return std::vector<double>(eigen_poles.data(),
                                   eigen_poles.data() + eigen_poles.size());
    }

    std::shared_ptr<sparseir::DiscreteLehmannRepresentation<S>> get_impl() const
    {
        return impl;
    }
};
