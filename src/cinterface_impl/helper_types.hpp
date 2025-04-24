#include "sparseir/sparseir.hpp"
#include <memory>

class AbstractFiniteTempBasis {
public:
    virtual ~AbstractFiniteTempBasis() = default;
    virtual int size() const = 0;
    virtual double get_beta() const = 0;
    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const = 0;
    virtual std::shared_ptr<AbstractContinuousFunctions> get_v() const = 0;
    virtual std::shared_ptr<AbstractMatsubaraFunctions> get_uhat() const = 0;
};

template<typename InternalType>
class FiniteTempBasis : public AbstractFiniteTempBasis {
private:
    std::shared_ptr<InternalType> impl;

public:
    FiniteTempBasis(std::shared_ptr<InternalType> impl): impl(impl) {}  

    virtual double get_beta() const override {
        return impl->get_beta();
    }

    virtual int size() const override {
        return impl->size();
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const override {
        return std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<ContinuousFunctions<InternalType>>(impl->get_u()));
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_v() const override {
        return std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<ContinuousFunctions<InternalType>>(impl->get_v()));
    }

    virtual std::shared_ptr<AbstractMatsubaraFunctions> get_uhat() const override {
        return std::static_pointer_cast<AbstractMatsubaraFunctions>(
            std::make_shared<MatsubaraBasisFunctions<InternalType>>(impl->get_uhat()));
    }
};

class AbstractMatsubaraFunctions {
public:
    virtual ~AbstractMatsubaraFunctions() = default;
    virtual Eigen::MatrixXcd operator()(const Eigen::ArrayXi &n_array) const = 0;
    virtual int size() const = 0;
};

template<typename S>
class MatsubaraBasisFunctions : public AbstractMatsubaraFunctions {
private:
    std::shared_ptr<sparseir::PiecewiseLegendreFTVector<S>> impl;

public:
    MatsubaraBasisFunctions(std::shared_ptr<sparseir::PiecewiseLegendreFTVector<S>> impl): impl(impl) {}

    virtual Eigen::MatrixXcd operator()(const Eigen::ArrayXi &n_array) const override {
        return impl->operator()(n_array);
    }

    virtual int size() const override {
        return impl->size();
    }
};


// Abstract class for functions of a single variable (in the imaginary time domain or the real frequency domain)
class AbstractContinuousFunctions {
public:
    virtual ~AbstractContinuousFunctions() = default;
    virtual Eigen::VectorXd operator()(double x) const = 0;
    virtual int size() const = 0;
};


template<typename InternalType>
class ContinuousFunctions : public AbstractContinuousFunctions {
private:
    std::shared_ptr<InternalType> impl;

public:
    ContinuousFunctions(std::shared_ptr<InternalType> impl): impl(impl) {}

    virtual Eigen::VectorXd operator()(double x) const override {
        return impl->operator()(x);
    }

    virtual int size() const override {
        return impl->size();
    }
};
