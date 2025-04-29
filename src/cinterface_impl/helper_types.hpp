#include "sparseir/sparseir.hpp"
#include <memory>


class AbstractMatsubaraFunctions {
public:
    virtual ~AbstractMatsubaraFunctions() = default;
    virtual Eigen::MatrixXcd operator()(const Eigen::ArrayXi &n_array) const = 0;
    virtual int size() const = 0;
};

template<typename InternalType>
class MatsubaraBasisFunctions : public AbstractMatsubaraFunctions {
private:
    std::shared_ptr<InternalType> impl;

public:
    MatsubaraBasisFunctions(std::shared_ptr<InternalType> impl): impl(impl) {}

    virtual Eigen::MatrixXcd operator()(const Eigen::ArrayXi &n_array) const override {
        return impl->operator()(n_array);
    }

    virtual int size() const override {
        return impl->size();
    }
};

// Abstract class for functions of a single variable 
class AbstractContinuousFunction {
public:
    virtual ~AbstractContinuousFunction() = default;
    virtual double operator()(double x) const = 0;
};


template<typename InternalType>
class ContinuousFunction : public AbstractContinuousFunction {
private:
    std::shared_ptr<InternalType> impl;

public:
    ContinuousFunction(std::shared_ptr<InternalType> impl): impl(impl) {}

    virtual double operator()(double x) const override {
        return impl->operator()(x);
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


//template<typename S>
//class DLRTauBasisFunctions : public AbstractContinuousFunctions {
//private:
    //std::shared_ptr<sparseir::TauPoles<S>> impl;
//
//public:
    //DLRTauBasisFunctions(std::shared_ptr<sparseir::TauPoles<S>> impl): impl(impl) {}
//
    //virtual Eigen::VectorXd operator()(double x) const override {
        //Eigen::VectorXd result(impl->size());
        //for (int i = 0; i < impl->size(); i++) {
            //result(i) = impl->operator()(x)(i);
        //}
        //return result;
    //}
//
    //virtual int size() const override {
        //return impl->size();
    //}
//};

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

template<typename S>
class _FiniteTempBasis : public AbstractFiniteTempBasis {
private:
    std::shared_ptr<sparseir::FiniteTempBasis<S>> impl;

public:
    _FiniteTempBasis(std::shared_ptr<sparseir::FiniteTempBasis<S>> impl): impl(impl) {}  

    virtual double get_beta() const override {
        return impl->get_beta();
    }

    virtual int size() const override {
        return impl->size();
    }

    virtual spir_statistics_type get_statistics() const override {
        if (std::is_same<S, sparseir::Fermionic>::value) {
            return SPIR_STATISTICS_FERMIONIC;
        } else {
            return SPIR_STATISTICS_BOSONIC;
        }
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const override {
        return std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<ContinuousFunctions<sparseir::PiecewiseLegendrePolyVector>>(impl->u));
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_v() const override {
        return std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<ContinuousFunctions<sparseir::PiecewiseLegendrePolyVector>>(impl->v));
    }

    virtual std::shared_ptr<AbstractMatsubaraFunctions> get_uhat() const override {
        return std::static_pointer_cast<AbstractMatsubaraFunctions>(
            std::make_shared<MatsubaraBasisFunctions<sparseir::PiecewiseLegendreFTVector<S>>>(impl->uhat));
    }

    std::shared_ptr<sparseir::FiniteTempBasis<S>> get_impl() const {
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

template<typename S>
class _DLR : public AbstractDLR {
private:
    std::shared_ptr<sparseir::DiscreteLehmannRepresentation<S>> impl;

public:
    _DLR(std::shared_ptr<sparseir::DiscreteLehmannRepresentation<S>> impl): impl(impl) {}

    virtual int size() const override {
        return impl->size();
    }

    virtual spir_statistics_type get_statistics() const override {
        if (std::is_same<S, sparseir::Fermionic>::value) {
            return SPIR_STATISTICS_FERMIONIC;
        } else {
            return SPIR_STATISTICS_BOSONIC;
        }
    }

    virtual double get_beta() const override {
        return impl->get_beta();
    }

    virtual std::shared_ptr<AbstractContinuousFunctions> get_u() const override {
        return std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<ContinuousFunctions<sparseir::TauPoles<S>>>(impl->u));
    }

    virtual std::shared_ptr<AbstractMatsubaraFunctions> get_uhat() const override {
        return std::static_pointer_cast<AbstractMatsubaraFunctions>(
            std::make_shared<MatsubaraBasisFunctions<sparseir::MatsubaraPoles<S>>>(impl->uhat));
    }

    virtual std::vector<double> get_poles() const override {
        Eigen::VectorXd eigen_poles = impl->poles;
        return std::vector<double>(eigen_poles.data(), eigen_poles.data() + eigen_poles.size());
    }

    std::shared_ptr<sparseir::DiscreteLehmannRepresentation<S>> get_impl() const {
        return impl;
    }
};
