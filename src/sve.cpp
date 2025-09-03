#include "sparseir/sve.hpp"
#include "sparseir/impl/sve_impl.hpp"

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

    int cut =
        std::count_if(s.data(), s.data() + s.size(),
                      [threshold](double val) { return val >= threshold; });

    if (max_size > 0) {
        cut = std::min(cut, max_size);
    }
    std::vector<PiecewiseLegendrePoly> u_part_(u->begin(), u->begin() + cut);
    PiecewiseLegendrePolyVector u_part(u_part_);
    Eigen::VectorXd s_part(s.head(cut));
    std::vector<PiecewiseLegendrePoly> v_part_(v->begin(), v->begin() + cut);
    PiecewiseLegendrePolyVector v_part(v_part_);
    return std::make_tuple(u_part, s_part, v_part);
}

std::tuple<double, std::string, std::string>
choose_accuracy(double epsilon, const std::string &Twork)
{
    using std::sqrt;

    if (Twork != "Float64" && Twork != "Float64x2" && Twork != "auto") {
        throw std::invalid_argument(
            "Twork must be either 'Float64', 'Float64x2', or 'auto'");
    }
    if (Twork == "Float64") {
        if (epsilon >= sqrt(std::numeric_limits<double>::epsilon())) {
            return std::make_tuple(epsilon, Twork, "default");
        } else {
            std::cerr << "Warning: Basis cutoff is " << epsilon
                      << ", which is below sqrt(ε) with numerical precision of double ε = "
                      << std::numeric_limits<double>::epsilon() << ".\n"
                      << "Expect singular values and basis functions for large "
                         "l to have lower precision than the cutoff.\n";
            return std::make_tuple(epsilon, Twork, "accurate");
        }
    } else if (Twork == "Float64x2") {
        if (epsilon >= sqrt(std::numeric_limits<xprec::DDouble>::epsilon())) {
            return std::make_tuple(epsilon, Twork, "default");
        } else {
            std::cerr << "Warning: Basis cutoff is " << epsilon
                      << ", which is below sqrt(ε) with numerical precision of double-double ε = "
                      << std::numeric_limits<xprec::DDouble>::epsilon() << ".\n"
                      << "Expect singular values and basis functions for large "
                         "l to have lower precision than the cutoff.\n";
            return std::make_tuple(epsilon, Twork, "accurate");
        }
    } else { // Twork == "auto"
        // Auto-select precision based on epsilon
        if (epsilon >= sqrt(std::numeric_limits<double>::epsilon())) {
            return std::make_tuple(epsilon, "Float64", "default");
        } else {
            std::cerr << "Warning: Basis cutoff is " << epsilon
                      << ", which is below sqrt(ε) with numerical precision of double-double ε = "
                      << std::numeric_limits<xprec::DDouble>::epsilon() << ".\n"
                      << "However, no higher precision arithmetic is available in the C API.\n";
            return std::make_tuple(epsilon, "Float64x2", "default");
        }
    }
}

std::tuple<double, std::string, std::string> choose_accuracy(double epsilon,
                                                             std::nullptr_t)
{
    if (epsilon >= std::sqrt(std::numeric_limits<double>::epsilon())) {
        return std::make_tuple(epsilon, "Float64", "default");
    } else {
        if (epsilon < std::sqrt(std::numeric_limits<double>::epsilon())) {
            std::cerr
                << "Warning: Basis cutoff is " << epsilon << ", which is"
                << " below sqrt(ε) with ε = "
                << std::numeric_limits<double>::epsilon() << ".\n"
                << "Expect singular values and basis functions for large l"
                << " to have lower precision than the cutoff.\n";
        }
        return std::make_tuple(epsilon, "Float64x2", "default");
    }
}

std::tuple<double, std::string, std::string> choose_accuracy(std::nullptr_t,
                                                             std::string Twork)
{
    if (Twork == "Float64x2") {
        const double epsilon = 2.220446049250313e-16;
        return std::make_tuple(epsilon, Twork, "default");
    } else {
        return std::make_tuple(
            std::sqrt(std::numeric_limits<double>::epsilon()), Twork,
            "default");
    }
}

std::tuple<double, std::string, std::string>
choose_accuracy_epsilon_nan(std::string Twork)
{
    if (Twork == "Float64x2") {
        const double epsilon = 2.220446049250313e-16;
        return std::make_tuple(epsilon, Twork, "default");
    } else {
        return std::make_tuple(
            std::sqrt(std::numeric_limits<double>::epsilon()), Twork,
            "default");
    }
}

std::tuple<double, std::string, std::string> choose_accuracy(std::nullptr_t,
                                                             std::nullptr_t)
{
    const double epsilon = 2.220446049250313e-16;
    return std::make_tuple(epsilon, "Float64x2", "default");
}

std::tuple<double, std::string, std::string>
auto_choose_accuracy(double epsilon, std::string Twork, std::string svd_strat)
{
    std::string auto_svd_strat;
    if (std::isnan(epsilon)) {
        std::tie(epsilon, Twork, auto_svd_strat) =
            choose_accuracy_epsilon_nan(Twork);
    } else {
        std::tie(epsilon, Twork, auto_svd_strat) =
            choose_accuracy(epsilon, Twork);
    }
    std::string final_svd_strat =
        (svd_strat == "auto") ? auto_svd_strat : svd_strat;
    return std::make_tuple(epsilon, Twork, final_svd_strat);
}

std::tuple<double, std::string, std::string>
auto_choose_accuracy(double epsilon, std::string Twork)
{
    return auto_choose_accuracy(epsilon, Twork, "auto");
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
                               std::string);
template SVEResult compute_sve(const RegularizedBoseKernel &, double, double,
                               int, int, std::string);

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
                      std::numeric_limits<int>::max(), -1, "Float64x2");
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