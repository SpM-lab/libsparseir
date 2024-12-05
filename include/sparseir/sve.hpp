// libsparseir/include/sparseir/sve.hpp

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <memory>

// Include other necessary headers here

namespace sparseir {

// Forward declarations
template <typename K>
class SVEResult;

std::tuple<double, std::string> choose_accuracy(double epsilon, const std::string& Twork, const std::string& svd_strat) {
    double safe_epsilon = epsilon;
    std::string Twork_actual = Twork;
    std::string svd_strategy_actual = svd_strat;

    if (epsilon >= std::sqrt(std::numeric_limits<double>::epsilon())) {
        svd_strategy_actual = "default";
    } else {
        std::cerr << "Warning: Basis cutoff is below √ε. Expect lower precision." << std::endl;
        svd_strategy_actual = "accurate";
    }

    return std::make_tuple(safe_epsilon, svd_strategy_actual);
}

// Base class for SVE strategies
template <typename K, typename T>
class AbstractSVE {
public:
    virtual ~AbstractSVE() {}
    virtual std::vector<Eigen::MatrixX<T>> matrices() const = 0;
    virtual SVEResult<K> postprocess(const std::vector<Eigen::MatrixX<T>>& u_list,
                                     const std::vector<Eigen::VectorX<T>>& s_list,
                                     const std::vector<Eigen::MatrixX<T>>& v_list) const = 0;

    int nsvals_hint;
};


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
    K kernel;
    T epsilon;
    int n_gauss;

    // Quadrature rules and segments
    Rule<T> rule;
    std::vector<T> segs_x;
    std::vector<T> segs_y;
    Rule<T> gauss_x;
    Rule<T> gauss_y;

    // Constructor
    SamplingSVE(const K& kernel_, T epsilon_, int n_gauss_ = -1)
        : AbstractSVE<K, T>(kernel_.sve_hints(epsilon_).nsvals_hint),
          kernel(kernel_), epsilon(epsilon_)
    {
        auto hints = kernel.sve_hints(epsilon);
        n_gauss = (n_gauss_ > 0) ? n_gauss_ : hints.ngauss;
        rule = GaussLegendreRule(n_gauss);
        segs_x = kernel.template segments_x<T>();
        segs_y = kernel.template segments_y<T>();
        gauss_x = GaussLegendreQuadrature(rule, segs_x);
        gauss_y = GaussLegendreQuadrature(rule, segs_y);
    }

    // Compute matrices for SVD
    std::vector<Eigen::MatrixXd> matrices() const override {
        std::vector<Eigen::MatrixXd> mats;

        size_t n_rows = gauss_x.points.size();
        size_t n_cols = gauss_y.points.size();

        Eigen::MatrixXd A(n_rows, n_cols);

        for (size_t i = 0; i < n_rows; ++i) {
            for (size_t j = 0; j < n_cols; ++j) {
                double x = gauss_x.points[i];
                double y = gauss_y.points[j];
                double wx = gauss_x.weights[i];
                double wy = gauss_y.weights[j];

                double K_xy = kernel.evaluate(x, y);
                A(i, j) = std::sqrt(wx) * K_xy * std::sqrt(wy);
            }
        }

        mats.push_back(A);
        return mats;
    }

    // Postprocess to construct SVEResult
    SVEResult<K> postprocess(
        const std::vector<Eigen::MatrixXd>& u_list,
        const std::vector<Eigen::VectorXd>& s_list,
        const std::vector<Eigen::MatrixXd>& v_list
    ) const override {
        // Assuming there's only one matrix in u_list, s_list, and v_list
        const auto& u_mat = u_list[0];
        const auto& s_vec = s_list[0];
        const auto& v_mat = v_list[0];

        // Number of segments
        size_t n_segments_x = segs_x.size() - 1;
        size_t n_segments_y = segs_y.size() - 1;

        // Compute the Legendre coefficients for u and v
        std::vector<Eigen::MatrixXd> u_data;
        std::vector<Eigen::MatrixXd> v_data;

        // For each segment, compute the Legendre coefficients
        for (size_t seg = 0; seg < n_segments_x; ++seg) {
            // Extract points and weights for the segment
            std::vector<double> x_points = gauss_x.points_segment(seg);
            std::vector<double> w_x = gauss_x.weights_segment(seg);

            // Collocation matrix for Legendre polynomials
            Eigen::MatrixXd cmat = legendre_collocation(rule);

            // Compute coefficients for each singular function
            Eigen::MatrixXd u_coeffs = cmat.colPivHouseholderQr().solve(u_mat);
            u_data.push_back(u_coeffs);
        }

        for (size_t seg = 0; seg < n_segments_y; ++seg) {
            std::vector<double> y_points = gauss_y.points_segment(seg);
            std::vector<double> w_y = gauss_y.weights_segment(seg);

            Eigen::MatrixXd cmat = legendre_collocation(rule);

            Eigen::MatrixXd v_coeffs = cmat.colPivHouseholderQr().solve(v_mat);
            v_data.push_back(v_coeffs);
        }

        // Construct PiecewiseLegendrePolyVector for u and v
        PiecewiseLegendrePolyVector u_pwv(u_data, segs_x);
        PiecewiseLegendrePolyVector v_pwv(v_data, segs_y);

        // Create SVEResult
        SVEResult<K> sve_result(u_pwv, s_vec, v_pwv, kernel, epsilon);

        return sve_result;
    }
};

// CentrosymmSVE class
template <typename K, typename T, typename InnerSVE = SamplingSVE<T, K>>
class CentrosymmSVE : public AbstractSVE<K, T> {
public:
    K kernel;
    T epsilon;
    InnerSVE even;
    InnerSVE odd;
    int nsvals_hint;

    CentrosymmSVE(const K& kernel_, T epsilon_, int n_gauss_ = -1)
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

    virtual SVEResult<K> postprocess(const std::vector<Eigen::MatrixX<T>>& u_list,
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
        auto segs_x_full = segments_x(full_hints);
        auto segs_y_full = segments_y(full_hints);

        std::vector<PiecewiseLegendrePoly> u_complete(u_sorted.size());
        std::vector<PiecewiseLegendrePoly> v_complete(v_sorted.size());

        Eigen::Array<T, Eigen::Dynamic, 1> poly_flip_x = Eigen::Array<T, Eigen::Dynamic, 1>::LinSpaced(u_sorted[0].data.rows(), 0, u_sorted[0].data.rows() - 1);
        poly_flip_x = poly_flip_x.unaryExpr([](T x) { return std::pow(-1, x); });

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
template <typename K>
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
template <typename K, typename T = double>
    auto compute_sve(K kernel,
                            T epsilon = std::numeric_limits<T>::quiet_NaN(),
                            T cutoff = std::numeric_limits<T>::quiet_NaN(),
                            int lmax = std::numeric_limits<int>::max(),
                            int n_gauss = -1,
                            const std::string& svd_strat = "auto") {
    // Choose accuracy parameters
    auto sve_strategy = SamplingSVE<K,T>(kernel, epsilon, n_gauss);

    double safe_epsilon;
    T Twork_actual;
    std::string svd_strategy_actual;
    // TODO: resolve choose_accuracy
    std::tie(safe_epsilon, Twork_actual, svd_strategy_actual) = choose_accuracy(epsilon, Twork, svd_strat);

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
    std::tuple<std::vector<Eigen::MatrixX<T>>, std::vector<Eigen::VectorX<T>>, std::vector<Eigen::MatrixX<T>>> truncated_result = truncate(u_list, s_list, v_list, cutoff_actual, lmax);
    std::vector<Eigen::MatrixX<T>> u_truncated = std::get<0>(truncated_result);
    std::vector<Eigen::VectorX<T>> s_truncated = std::get<1>(truncated_result);
    std::vector<Eigen::MatrixX<T>> v_truncated = std::get<2>(truncated_result);

    // Postprocess to get the SVEResult
    return sve_strategy.postprocess(u_truncated, s_truncated, v_truncated);
}

// Function to truncate singular values
inline void truncate_singular_values(
    std::vector<Eigen::MatrixXd>& u_list,
    std::vector<Eigen::VectorXd>& s_list,
    std::vector<Eigen::MatrixXd>& v_list,
    double rtol,
    int lmax
) {
    // Collect all singular values
    std::vector<double> all_singular_values;
    for (const auto& s : s_list) {
        for (int i = 0; i < s.size(); ++i) {
            all_singular_values.push_back(s(i));
        }
    }
    std::sort(all_singular_values.begin(), all_singular_values.end(), std::greater<double>());

    // Determine cutoff
    double cutoff = rtol * all_singular_values.front();
    if (lmax < static_cast<int>(all_singular_values.size())) {
        cutoff = std::max(cutoff, all_singular_values[lmax - 1]);
    }

    // Truncate singular values and corresponding vectors
    for (size_t idx = 0; idx < s_list.size(); ++idx) {
        const auto& s = s_list[idx];
        int scount = 0;
        for (int i = 0; i < s.size(); ++i) {
            if (s(i) > cutoff) {
                ++scount;
            } else {
                break;
            }
        }
        if (scount < s.size()) {
            u_list[idx] = u_list[idx].leftCols(scount);
            s_list[idx] = s_list[idx].head(scount);
            v_list[idx] = v_list[idx].leftCols(scount);
        }
    }
}


// Function to canonicalize basis functions
inline void canonicalize(
    PiecewiseLegendrePolyVector& u,
    PiecewiseLegendrePolyVector& v
) {
    for (size_t i = 0; i < u.size(); ++i) {
        double gauge = std::copysign(1.0, u.polyvec[i](1.0));
        u.polyvec[i].data *= gauge;
        v.polyvec[i].data *= gauge;
    }
}

} // namespace sparseir