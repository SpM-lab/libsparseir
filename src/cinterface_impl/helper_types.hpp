#include "sparseir/sparseir.hpp"
#include <memory>


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
