#pragma once

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <memory>

// Forward declarations of types and functions used (to be defined elsewhere)
template <typename T>
struct Rule;

template <typename T>
Rule<T> legendre(int n);

template <typename T>
std::vector<T> segments_x(const SveHints& hints);

template <typename T>
std::vector<T> segments_y(const SveHints& hints);

template <typename T>
Rule<T> piecewise(const Rule<T>& rule, const std::vector<T>& segments);

template <typename K, typename T>
Eigen::MatrixX<T> matrix_from_gauss(const K& kernel, const Rule<T>& gauss_x, const Rule<T>& gauss_y);

template <typename T>
Eigen::MatrixX<T> legendre_collocation(const Rule<T>& rule);

void canonicalize(PiecewiseLegendrePolyVector& ulx, PiecewiseLegendrePolyVector& vly);

// Namespace
namespace sparseir {

// Base class for SVE strategies
template <typename T>
class AbstractSVE {
public:
    virtual ~AbstractSVE() {}
    virtual std::vector<Eigen::MatrixX<T>> matrices() const = 0;
<<<<<<< HEAD
    virtual SVEResult<K> postprocess(
        const std::vector<Eigen::MatrixX<T>>& u_list,
        const std::vector<Eigen::VectorX<T>>& s_list,
        const std::vector<Eigen::MatrixX<T>>& v_list
    ) const = 0;
=======
    virtual SVEResult<T> postprocess(const std::vector<Eigen::MatrixX<T>>& u_list,
                                     const std::vector<Eigen::VectorX<T>>& s_list,
                                     const std::vector<Eigen::MatrixX<T>>& v_list) const = 0;
>>>>>>> 238cbe6 (TODO: resolve errors on SVEResult)

    int nsvals_hint;
};

<<<<<<< HEAD

// SamplingSVE class
/**
 * @brief Sampling-based SVE to SVD translation.
 *
 * Maps the singular value expansion (SVE) of a kernel onto the singular
 * value decomposition (SVD) of a matrix by choosing two sets of Gauss
 * quadrature rules and approximating the integrals in the SVE equations
 * by finite sums.
 */
template <typename K, typename T = double>
class SamplingSVE : public AbstractSVE<K, T> {
public:
    std::shared_ptr<const K> kernel;
    double epsilon;
    int n_gauss;
=======
// SamplingSVE class
template <typename T, typename K>
class SamplingSVE : public AbstractSVE<T> {
public:
    K kernel;
    double epsilon;
    int n_gauss;
    int nsvals_hint;
>>>>>>> 238cbe6 (TODO: resolve errors on SVEResult)

    // Quadrature rules and segments
    Rule<T> rule;
    std::vector<T> segs_x;
    std::vector<T> segs_y;
    Rule<T> gauss_x;
    Rule<T> gauss_y;

    SamplingSVE(const K& kernel_, double epsilon_, int n_gauss_ = -1)
        : kernel(kernel_), epsilon(epsilon_) {
        auto sve_hints_ = sve_hints(kernel, epsilon);
        n_gauss = (n_gauss_ > 0) ? n_gauss_ : ngauss(sve_hints_);
        rule = legendre<T>(n_gauss);
        segs_x = segments_x<T>(sve_hints_);
        segs_y = segments_y<T>(sve_hints_);
        gauss_x = piecewise(rule, segs_x);
        gauss_y = piecewise(rule, segs_y);
        nsvals_hint = nsvals(sve_hints_);
    }

    virtual std::vector<Eigen::MatrixX<T>> matrices() const override {
        Eigen::MatrixX<T> result = matrix_from_gauss(kernel, gauss_x, gauss_y);
        result = result.array().colwise() * gauss_y.w.array().sqrt();
        result = result.array().rowwise() * gauss_x.w.array().sqrt().transpose();
        return { result };
    }

    virtual SVEResult<T> postprocess(const std::vector<Eigen::MatrixX<T>>& u_list,
                                     const std::vector<Eigen::VectorX<T>>& s_list,
                                     const std::vector<Eigen::MatrixX<T>>& v_list) const override {
        const auto& u = u_list[0];
        const auto& s = s_list[0];
        const auto& v = v_list[0];

        Eigen::MatrixX<T> u_x = u.array().colwise() / gauss_x.w.array().sqrt();
        Eigen::MatrixX<T> v_y = v.array().colwise() / gauss_y.w.array().sqrt();

        int num_segments_x = static_cast<int>(segs_x.size()) - 1;
        int num_segments_y = static_cast<int>(segs_y.size()) - 1;

        u_x.resize(n_gauss, num_segments_x * s.size());
        v_y.resize(n_gauss, num_segments_y * s.size());

        Eigen::MatrixX<T> cmat = legendre_collocation(rule);
        Eigen::MatrixX<T> u_data = cmat * u_x;
        Eigen::MatrixX<T> v_data = cmat * v_y;

        Eigen::Array<T, Eigen::Dynamic, 1> dsegs_x = (Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(segs_x.data(), segs_x.size() - 1)).template cast<T>().diff();
        Eigen::Array<T, Eigen::Dynamic, 1> dsegs_y = (Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(segs_y.data(), segs_y.size() - 1)).template cast<T>().diff();

        u_data.array().rowwise() *= (dsegs_x * T(0.5)).sqrt().transpose();
        v_data.array().rowwise() *= (dsegs_y * T(0.5)).sqrt().transpose();

        // Construct polynomials
        PiecewiseLegendrePolyVector ulx(u_data, segs_x);
        PiecewiseLegendrePolyVector vly(v_data, segs_y);
        canonicalize(ulx, vly);

        return SVEResult<T>(ulx, s.template cast<double>(), vly, kernel, epsilon);
    }
};

// CentrosymmSVE class
template <typename T, typename K, typename InnerSVE = SamplingSVE<T, K>>
class CentrosymmSVE : public AbstractSVE<T> {
public:
    K kernel;
    double epsilon;
    InnerSVE even;
    InnerSVE odd;
    int nsvals_hint;

    CentrosymmSVE(const K& kernel_, double epsilon_, int n_gauss_ = -1)
        : kernel(kernel_), epsilon(epsilon_),
          even(get_symmetrized(kernel, +1), epsilon_, n_gauss_),
          odd(get_symmetrized(kernel, -1), epsilon_, n_gauss_) {
        nsvals_hint = std::max(even.nsvals_hint, odd.nsvals_hint);
    }

    virtual std::vector<Eigen::MatrixX<T>> matrices() const override {
        auto mats_even = even.matrices();
        auto mats_odd = odd.matrices();
        return { mats_even[0], mats_odd[0] };
    }

    virtual SVEResult<T> postprocess(const std::vector<Eigen::MatrixX<T>>& u_list,
                                     const std::vector<Eigen::VectorX<T>>& s_list,
                                     const std::vector<Eigen::MatrixX<T>>& v_list) const override {
        SVEResult<T> result_even = even.postprocess({ u_list[0] }, { s_list[0] }, { v_list[0] });
        SVEResult<T> result_odd = odd.postprocess({ u_list[1] }, { s_list[1] }, { v_list[1] });

        // Merge results
        auto u = result_even.u;
        u.insert(u.end(), result_odd.u.begin(), result_odd.u.end());
        auto s = result_even.s;
        s.insert(s.end(), result_odd.s.begin(), result_odd.s.end());
        auto v = result_even.v;
        v.insert(v.end(), result_odd.v.begin(), result_odd.v.end());

        std::vector<int> signs(result_even.s.size(), +1);
        signs.insert(signs.end(), result_odd.s.size(), -1);

        // Sort singular values and associated vectors
        std::vector<size_t> indices(s.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(), [&](size_t i1, size_t i2) {
            return s[i1] > s[i2];
        });

        std::vector<PiecewiseLegendrePoly> u_sorted(u.size());
        std::vector<double> s_sorted(s.size());
        std::vector<PiecewiseLegendrePoly> v_sorted(v.size());
        std::vector<int> signs_sorted(signs.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            u_sorted[i] = u[indices[i]];
            s_sorted[i] = s[indices[i]];
            v_sorted[i] = v[indices[i]];
            signs_sorted[i] = signs[indices[i]];
        }

        // Extend to negative side
        // Assuming definitions of necessary functions and data structures
        auto full_hints = sve_hints(kernel, epsilon);
        auto segs_x_full = segments_x<T>(full_hints);
        auto segs_y_full = segments_y<T>(full_hints);

        std::vector<PiecewiseLegendrePoly> u_complete(u_sorted.size());
        std::vector<PiecewiseLegendrePoly> v_complete(v_sorted.size());

        Eigen::Array<T, Eigen::Dynamic, 1> poly_flip_x = Eigen::Array<T, Eigen::Dynamic, 1>::LinSpaced(u_sorted[0].data.rows(), T(0), T(u_sorted[0].data.rows() - 1));
        poly_flip_x = (-1).pow(poly_flip_x);

        for (size_t i = 0; i < u_sorted.size(); ++i) {
            Eigen::MatrixX<T> u_pos_data = u_sorted[i].data / std::sqrt(T(2));
            Eigen::MatrixX<T> v_pos_data = v_sorted[i].data / std::sqrt(T(2));

            Eigen::MatrixX<T> u_neg_data = u_pos_data.rowwise().reverse().array().colwise() * poly_flip_x.array() * T(signs_sorted[i]);
            Eigen::MatrixX<T> v_neg_data = v_pos_data.rowwise().reverse().array().colwise() * poly_flip_x.array() * T(signs_sorted[i]);

            Eigen::MatrixX<T> u_data_full(u_pos_data.rows(), u_pos_data.cols() + u_neg_data.cols());
            u_data_full << u_neg_data, u_pos_data;

            Eigen::MatrixX<T> v_data_full(v_pos_data.rows(), v_pos_data.cols() + v_neg_data.cols());
            v_data_full << v_neg_data, v_pos_data;

            u_complete[i] = PiecewiseLegendrePoly(u_data_full, segs_x_full, static_cast<int>(i), signs_sorted[i]);
            v_complete[i] = PiecewiseLegendrePoly(v_data_full, segs_y_full, static_cast<int>(i), signs_sorted[i]);
        }

        return SVEResult<T>(u_complete, s_sorted, v_complete, kernel, epsilon);
    }
};

// SVEResult class
template <typename T, typename K>
class SVEResult {
public:
    PiecewiseLegendrePolyVector u;
    std::vector<double> s;
    PiecewiseLegendrePolyVector v;

    K kernel;
    double epsilon;

    SVEResult(const PiecewiseLegendrePolyVector& u_, const std::vector<double>& s_,
              const PiecewiseLegendrePolyVector& v_, const K& kernel_, double epsilon_)
        : u(u_), s(s_), v(v_), kernel(kernel_), epsilon(epsilon_) {}
};

// Function to compute SVE result
template <typename T, typename K, typename SVEstrategy = SamplingSVE<T, K>>
SVEResult<T, K> compute_sve(const K& kernel,
                            double epsilon = std::numeric_limits<double>::quiet_NaN(),
                            double cutoff = std::numeric_limits<double>::quiet_NaN(),
                            int lmax = std::numeric_limits<int>::max(),
                            int n_gauss = -1,
                            const std::string& svd_strat = "auto",
                            const SVEstrategy& sve_strategy = SVEstrategy(kernel, epsilon, n_gauss)) {
    // Choose accuracy parameters
    double safe_epsilon;
    T Twork_actual;
    std::string svd_strategy_actual;
    std::tie(safe_epsilon, Twork_actual, svd_strategy_actual) = choose_accuracy(epsilon, Twork_actual, svd_strat);

    // Compute SVDs
    auto matrices = sve_strategy.matrices();
    std::vector<Eigen::BDCSVD<Eigen::MatrixX<T>>> svds;
    for (const auto& mat : matrices) {
        Eigen::BDCSVD<Eigen::MatrixX<T>> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        svds.push_back(svd);
    }

    // Extract singular values and vectors
    std::vector<Eigen::MatrixX<T>> u_list, v_list;
    std::vector<Eigen::VectorX<T>> s_list;
    for (const auto& svd : svds) {
        u_list.push_back(svd.matrixU());
        s_list.push_back(svd.singularValues());
        v_list.push_back(svd.matrixV());
    }

    // Apply cutoff and lmax
    double cutoff_actual = std::isnan(cutoff) ? 2 * std::numeric_limits<T>::epsilon() : cutoff;
    auto [u_truncated, s_truncated, v_truncated] = truncate(u_list, s_list, v_list, cutoff_actual, lmax);

    // Postprocess to get the SVEResult
    return sve_strategy.postprocess(u_truncated, s_truncated, v_truncated);
}

// Helper functions
template <typename T>
std::tuple<double, T, std::string>
choose_accuracy(double epsilon, T Twork, const std::string& svd_strat) {
    double safe_epsilon = std::isnan(epsilon) ? std::sqrt(std::numeric_limits<T>::epsilon()) : epsilon;
    Twork = Twork; // Assuming Twork is provided or deduced
    std::string auto_svd_strat = (svd_strat == "auto") ? "default" : svd_strat;
    return std::make_tuple(safe_epsilon, Twork, auto_svd_strat);
}

template <typename T>
std::tuple<std::vector<Eigen::MatrixX<T>>, std::vector<Eigen::VectorX<T>>, std::vector<Eigen::MatrixX<T>>>
truncate(const std::vector<Eigen::MatrixX<T>>& u_list,
         const std::vector<Eigen::VectorX<T>>& s_list,
         const std::vector<Eigen::MatrixX<T>>& v_list,
         double rtol = 0.0,
         int lmax = std::numeric_limits<int>::max()) {
    // Collect all singular values
    std::vector<T> s_all;
    for (const auto& s : s_list) {
        s_all.insert(s_all.end(), s.data(), s.data() + s.size());
    }
    // Sort and determine cutoff
    std::sort(s_all.begin(), s_all.end(), std::greater<T>());
    T cutoff = (lmax < static_cast<int>(s_all.size())) ? std::max(rtol * s_all.front(), s_all[lmax - 1]) : rtol * s_all.front();

    // Truncate singular values and associated vectors
    std::vector<Eigen::MatrixX<T>> u_truncated;
    std::vector<Eigen::VectorX<T>> s_truncated;
    std::vector<Eigen::MatrixX<T>> v_truncated;

    for (size_t i = 0; i < s_list.size(); ++i) {
        const auto& s = s_list[i];
        int count = (int)std::count_if(s.data(), s.data() + s.size(),
                                       [cutoff](T val) { return val > cutoff; });
        if (count > 0) {
            u_truncated.push_back(u_list[i].leftCols(count));
            s_truncated.push_back(s.head(count));
            v_truncated.push_back(v_list[i].leftCols(count));
        }
    }
    return std::make_tuple(u_truncated, s_truncated, v_truncated);
}

// Function to canonicalize u and v
void canonicalize(PiecewiseLegendrePolyVector& ulx, PiecewiseLegendrePolyVector& vly) {
    // Implement canonicalization based on u(1) > 0
    for (size_t i = 0; i < ulx.size(); ++i) {
        double gauge = std::copysign(1.0, ulx[i](1.0));
        ulx[i].data *= gauge;
        vly[i].data *= gauge;
    }
}

// Additional helper functions and definitions would be required to complete this implementation
// such as PiecewiseLegendrePoly, sve_hints, ngauss, nsvals, get_symmetrized, etc.

} // namespace sparseir