#include "sparseir/sve.hpp"
#include "sparseir/impl/sve_impl.ipp"

namespace sparseir {

using xprec::DDouble;

SVEResult::SVEResult() { }

SVEResult::SVEResult(const PiecewiseLegendrePolyVector &u_,
                     const Eigen::VectorXd &s_,
                     const PiecewiseLegendrePolyVector &v_, double epsilon_)
    : u(std::make_shared<PiecewiseLegendrePolyVector>(u_)),
      s(s_),
      v(std::make_shared<PiecewiseLegendrePolyVector>(v_)),
      epsilon(epsilon_)
{
}

std::tuple<PiecewiseLegendrePolyVector, Eigen::VectorXd,
           PiecewiseLegendrePolyVector>
SVEResult::part(double eps, int max_size) const
{
    eps = std::isnan(eps) ? this->epsilon : eps;
    double threshold = eps * s(0);

    auto cut =
        std::count_if(s.data(), s.data() + s.size(),
                      [threshold](double val) { return val >= threshold; });

    if (max_size > 0) {
        cut = std::min(cut, static_cast<decltype(cut)>(max_size));
    }
    std::vector<PiecewiseLegendrePoly> u_part_(u->begin(), u->begin() + cut);
    PiecewiseLegendrePolyVector u_part(u_part_);
    Eigen::VectorXd s_part(s.head(cut));
    std::vector<PiecewiseLegendrePoly> v_part_(v->begin(), v->begin() + cut);
    PiecewiseLegendrePolyVector v_part(v_part_);
    return std::make_tuple(u_part, s_part, v_part);
}

std::tuple<double, TworkType, SVDStrategy>
safe_epsilon(double eps_required, TworkType work_dtype, SVDStrategy svd_strat)
{
    using std::sqrt;

    if (eps_required < 0.0) {
        throw std::runtime_error("eps_required must be non-negative");
    }
    
    // First, choose the working dtype based on the eps required
    TworkType actual_work_dtype;
    if (work_dtype == TworkType::AUTO) {
        if (std::isnan(eps_required) || eps_required < 1e-8) {
            actual_work_dtype = TworkType::FLOAT64X2;  // MAX_DTYPE equivalent
        } else {
            actual_work_dtype = TworkType::FLOAT64;
        }
    } else {
        actual_work_dtype = work_dtype;
    }

    // Next, work out the actual epsilon
    double safe_eps;
    if (actual_work_dtype == TworkType::FLOAT64) {
        // This is technically a bit too low (the true value is about 1.5e-8),
        // but it's not too far off and easier to remember for the user.
        safe_eps = 1e-8;
    } else {
        safe_eps = static_cast<double>(sqrt(std::numeric_limits<xprec::DDouble>::epsilon()));
    }

    // Work out the SVD strategy to be used. If the user sets this, we
    // assume they know what they are doing and do not warn if they compute
    // the basis.
    bool warn_acc = false;
    SVDStrategy actual_svd_strat;
    if (svd_strat == SVDStrategy::AUTO) {
        if (!std::isnan(eps_required) && eps_required < safe_eps) {
            actual_svd_strat = SVDStrategy::ACCURATE;
            warn_acc = true;
        } else {
            actual_svd_strat = SVDStrategy::FAST;
        }
    } else {
        actual_svd_strat = svd_strat;
    }

    if (warn_acc) {
        std::cerr << "\nRequested accuracy is " << eps_required
                  << ", which is below the\naccuracy " << safe_eps
                  << " for the work data type " << (actual_work_dtype == TworkType::FLOAT64 ? "float64" : "float64x2")
                  << ".\nExpect singular values and basis functions for large l to\n"
                  << "have lower precision than the cutoff.\n";
    }

    return std::make_tuple(safe_eps, actual_work_dtype, actual_svd_strat);
}


void canonicalize(PiecewiseLegendrePolyVector &u,
                  PiecewiseLegendrePolyVector &v)
{
    for (size_t i = 0; i < u.size(); ++i) {
        double gauge = std::copysign(1.0, u.polyvec[i](1.0));
        u.polyvec[i].data *= gauge;
        v.polyvec[i].data *= gauge;
    }
}

// Explicit template instantiations
template SVEResult compute_sve(const LogisticKernel &, double, double, int, int,
                               TworkType);
template SVEResult compute_sve(const RegularizedBoseKernel &, double, double,
                               int, int, TworkType);

template class SamplingSVE<LogisticKernel, double>;
template class SamplingSVE<LogisticKernel, DDouble>;
template class SamplingSVE<RegularizedBoseKernel, double>;
template class SamplingSVE<RegularizedBoseKernel, DDouble>;
template class SamplingSVE<ReducedKernel<LogisticKernel>, double>;
template class SamplingSVE<ReducedKernel<LogisticKernel>, DDouble>;
template class SamplingSVE<LogisticKernelOdd, double>;
template class SamplingSVE<LogisticKernelOdd, DDouble>;

// 2-argument version of compute_sve
template <typename K>
SVEResult compute_sve(const K &kernel, double epsilon)
{
    return compute_sve(kernel, epsilon, std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<int>::max(), -1, TworkType::FLOAT64X2);
}

// SVEParams version of compute_sve
template <typename K>
SVEResult compute_sve(const K &kernel, double epsilon, const SVEParams& params)
{
    return compute_sve(kernel, epsilon, params.cutoff, params.lmax, params.n_gauss, params.Twork);
}

// Explicit template instantiations for SVEParams version
template SVEResult compute_sve<LogisticKernel>(const LogisticKernel &kernel, double epsilon, const SVEParams& params);
template SVEResult compute_sve<RegularizedBoseKernel>(const RegularizedBoseKernel &kernel, double epsilon, const SVEParams& params);

// Explicit template instantiations for CentrosymmSVE
template class CentrosymmSVE<LogisticKernel, double>;
template class CentrosymmSVE<LogisticKernel, DDouble>;
template class CentrosymmSVE<RegularizedBoseKernel, double>;
template class CentrosymmSVE<RegularizedBoseKernel, DDouble>;

template std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>,
                    std::vector<Eigen::MatrixXd>>
truncate(const std::vector<Eigen::MatrixXd> &u,
         const std::vector<Eigen::VectorXd> &s,
         const std::vector<Eigen::MatrixXd> &v, double rtol, int lmax);

template std::tuple<
    std::vector<Eigen::Matrix<DDouble, Eigen::Dynamic, Eigen::Dynamic>>,
    std::vector<Eigen::Vector<DDouble, Eigen::Dynamic>>,
    std::vector<Eigen::Matrix<DDouble, Eigen::Dynamic, Eigen::Dynamic>>>
truncate(
    const std::vector<Eigen::Matrix<DDouble, Eigen::Dynamic, Eigen::Dynamic>>
        &u,
    const std::vector<Eigen::Vector<DDouble, Eigen::Dynamic>> &s,
    const std::vector<Eigen::Matrix<DDouble, Eigen::Dynamic, Eigen::Dynamic>>
        &v,
    DDouble rtol, int lmax);

} // namespace sparseir
