#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "sparseir/sparseir-fwd.hpp"
#include "sparseir/gauss.hpp"
#include "sparseir/poly.hpp"
#include "sparseir/kernel.hpp"
#include "sparseir/impl/kernel_impl.ipp"
#include "sparseir/svd.hpp"
namespace sparseir {

// Twork type enum to match C API constants
enum class TworkType {
    FLOAT64 = 0,     // SPIR_TWORK_FLOAT64
    FLOAT64X2 = 1,   // SPIR_TWORK_FLOAT64X2
    AUTO = -1        // SPIR_TWORK_AUTO
};


enum class SVDStrategy {
    FAST = 0,   // SPIR_SVDSTRAT_FAST
    ACCURATE = 1,  // SPIR_SVDSTRAT_ACCURATE
    AUTO = -1     // SPIR_SVDSTRAT_AUTO
};

class SVEResult {
public:
    std::shared_ptr<PiecewiseLegendrePolyVector> u;
    Eigen::VectorXd s;
    std::shared_ptr<PiecewiseLegendrePolyVector> v;
    double epsilon;

    SVEResult();
    SVEResult(const PiecewiseLegendrePolyVector &u_, const Eigen::VectorXd &s_,
              const PiecewiseLegendrePolyVector &v_, double epsilon_);

    std::tuple<PiecewiseLegendrePolyVector, Eigen::VectorXd,
               PiecewiseLegendrePolyVector>
    part(double eps = std::numeric_limits<double>::quiet_NaN(),
         int max_size = -1) const;
};

std::tuple<double, TworkType, SVDStrategy>
safe_epsilon(
    double epsilon = std::numeric_limits<double>::quiet_NaN(),
    TworkType Twork = TworkType::AUTO,
    SVDStrategy svd_strat = SVDStrategy::AUTO
);

void canonicalize(PiecewiseLegendrePolyVector &u,
                  PiecewiseLegendrePolyVector &v);

// Base class for SVE strategies
template <typename T>
class AbstractSVE {
public:
    virtual ~AbstractSVE() { }
    virtual std::vector<Eigen::MatrixX<T>> matrices() const = 0;
    virtual SVEResult
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const = 0;
};

// SamplingSVE class declaration
template <typename K, typename T>
class SamplingSVE : public AbstractSVE<T> {
public:
    std::shared_ptr<AbstractKernel> kernel;
    double epsilon;
    int n_gauss;
    int nsvals_hint;
    Rule<T> rule;
    std::vector<T> segs_x;
    std::vector<T> segs_y;
    Rule<T> gauss_x;
    Rule<T> gauss_y;

    SamplingSVE(const K &kernel_, double epsilon_, int n_gauss_ = -1);
    SamplingSVE(const std::shared_ptr<AbstractKernel> &kernel_, double epsilon_,
                int n_gauss_ = -1);

    std::vector<Eigen::MatrixX<T>> matrices() const override;
    SVEResult
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const override;
};

// CentrosymmSVE class declaration
template <typename K, typename T>
class CentrosymmSVE : public AbstractSVE<T> {
public:
    std::shared_ptr<AbstractKernel> kernel;
    double epsilon;
    SamplingSVE<
        typename SymmKernelTraits<K, std::integral_constant<int, +1>>::type, T>
        even;
    SamplingSVE<
        typename SymmKernelTraits<K, std::integral_constant<int, -1>>::type, T>
        odd;
    int nsvals_hint;

    CentrosymmSVE(const K &kernel_, double epsilon_, int n_gauss_ = -1);
    CentrosymmSVE(const std::shared_ptr<AbstractKernel> &kernel_,
                  double epsilon_, int n_gauss_ = -1);

    std::vector<Eigen::MatrixX<T>> matrices() const override;
    SVEResult
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const override;
};


// Template function declarations
template <typename K, typename T>
std::shared_ptr<AbstractSVE<T>> determine_sve(const K &kernel,
                                              double safe_epsilon, int n_gauss);

//template <typename T>
//std::shared_ptr<AbstractSVE<T>>
//determine_sve(const std::shared_ptr<AbstractKernel> &kernel,
              //double safe_epsilon, int n_gauss);

template <typename T>
std::tuple<std::vector<Eigen::MatrixX<T>>, std::vector<Eigen::VectorX<T>>,
           std::vector<Eigen::MatrixX<T>>>
truncate(const std::vector<Eigen::MatrixX<T>> &u,
         const std::vector<Eigen::VectorX<T>> &s,
         const std::vector<Eigen::MatrixX<T>> &v, T rtol = 0.0,
         int lmax = std::numeric_limits<int>::max());

template <typename K, typename T>
std::tuple<SVEResult, std::shared_ptr<AbstractSVE<T>>>
pre_postprocess(const K &kernel, double safe_epsilon, int n_gauss,
                double cutoff = std::numeric_limits<double>::quiet_NaN(),
                int lmax = std::numeric_limits<int>::max());

// SVE computation parameters
struct SVEParams {
    double cutoff = std::numeric_limits<double>::quiet_NaN();
    int lmax = std::numeric_limits<int>::max();
    int n_gauss = -1;
    TworkType Twork = TworkType::AUTO;
    
    // Default constructor with default values
    SVEParams() = default;
    
    // Constructor with custom values
    SVEParams(double cutoff_, int lmax_, int n_gauss_, TworkType Twork_)
        : cutoff(cutoff_), lmax(lmax_), n_gauss(n_gauss_), Twork(Twork_) {}
    
    // Constructor with custom cutoff only
    explicit SVEParams(double cutoff_) : cutoff(cutoff_) {}
    
    // Constructor with custom Twork only
    explicit SVEParams(TworkType Twork_) : Twork(Twork_) {}
};

// Restrict compute_sve to concrete kernel types only
template <typename K>
SVEResult compute_sve(const K &kernel, double epsilon,
            double cutoff = std::numeric_limits<double>::quiet_NaN(),
            int lmax = std::numeric_limits<int>::max(),
            int n_gauss = -1, TworkType Twork = TworkType::AUTO);

// New overload using SVEParams struct
template <typename K>
SVEResult compute_sve(const K &kernel, double epsilon, const SVEParams& params);


} // namespace sparseir
