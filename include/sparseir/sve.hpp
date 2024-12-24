// libsparseir/include/sparseir/sve.hpp

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor> // Include Tensor module
#include <vector>

// Include other necessary headers here

namespace sparseir {

// Forward declarations
template <typename K>
class SVEResult;

inline std::tuple<double, std::string, std::string>
choose_accuracy(double epsilon, std::string Twork)
{
    if (Twork == "Float64") {
        if (epsilon >= std::sqrt(std::numeric_limits<double>::epsilon())) {
            return std::make_tuple(epsilon, Twork, "default");
        } else {
            std::cerr << "Warning: Basis cutoff is " << epsilon
                      << ", which is below √ε with ε = "
                      << std::numeric_limits<double>::epsilon() << ".\n"
                      << "Expect singular values and basis functions for large "
                         "l to have lower precision than the cutoff.\n";
            return std::make_tuple(epsilon, Twork, "accurate");
        }
    } else {
        // Handle the case for xprec::DDouble
        if (epsilon >= std::sqrt(std::numeric_limits<double>::epsilon())) {
            return std::make_tuple(epsilon, Twork, "default");
        } else {
            std::cerr << "Warning: Basis cutoff is " << epsilon
                      << ", which is below √ε with ε = "
                      << std::numeric_limits<xprec::DDouble>::epsilon() << ".\n"
                      << "Expect singular values and basis functions for large "
                         "l to have lower precision than the cutoff.\n";
            return std::make_tuple(epsilon, Twork, "accurate");
        }
    }
}

inline std::tuple<double, std::string, std::string>
choose_accuracy(double epsilon, std::nullptr_t)
{
    if (epsilon >= std::sqrt(std::numeric_limits<double>::epsilon())) {
        return std::make_tuple(epsilon, "Float64", "default");
    } else {
        // This should work, but catch2 can't catch this warning
        // Therefore we suppress this block
        if (epsilon < std::sqrt(std::numeric_limits<double>::epsilon())) {
            std::cerr << "Warning: Basis cutoff is " << epsilon << ", which is"
                      << " below √ε with ε = "
                      << std::numeric_limits<double>::epsilon() << ".\n"
                      << "Expect singular values and basis functions for large l"
                      << " to have lower precision than the cutoff.\n";
        }
        return std::make_tuple(epsilon, "Float64x2", "default");
    }
}

// Equivalent to Julia implementation:
// julia> choose_accuracy(::Nothing, Twork) = sqrt(eps(Twork)), Twork, :default
inline std::tuple<double, std::string, std::string>
choose_accuracy(std::nullptr_t, std::string Twork)
{
    if (Twork == "Float64x2") {
        const double epsilon =
            2.220446049250313e-16; // julia> using MultiFloats;
                                   // Float64(sqrt(eps(Float64x2)))
        return std::make_tuple(epsilon, Twork, "default");
    } else {
        return std::make_tuple(
            std::sqrt(std::numeric_limits<double>::epsilon()), Twork,
            "default");
    }
}

inline std::tuple<double, std::string, std::string>
choose_accuracy_epsilon_nan(std::string Twork)
{
    if (Twork == "Float64x2") {
        const double epsilon =
            2.220446049250313e-16; // julia> using MultiFloats;
                                   // Float64(sqrt(eps(Float64x2)))
        return std::make_tuple(epsilon, Twork, "default");
    } else {
        return std::make_tuple(
            std::sqrt(std::numeric_limits<double>::epsilon()), Twork,
            "default");
    }
}

// Equivalent to Julia implementation:
// julia> choose_accuracy(::Nothing, ::Nothing) = Float64(sqrt(eps(T_MAX))),
// T_MAX, :default
inline std::tuple<double, std::string, std::string>
choose_accuracy(std::nullptr_t, std::nullptr_t)
{
    const double epsilon =
        2.220446049250313e-16; // julia> using MultiFloats;
                               // Float64(sqrt(eps(Float64x2)))
    return std::make_tuple(epsilon, "Float64x2", "default");
}

inline std::tuple<double, std::string, std::string>
choose_accuracy(double epsilon, std::string Twork, std::string svd_strat)
{
    std::string auto_svd_strat;
    if (std::isnan(epsilon)) {
        std::tie(epsilon, Twork, auto_svd_strat) =
            choose_accuracy_epsilon_nan(Twork);
    } else {
        std::tie(epsilon, Twork, auto_svd_strat) =
            choose_accuracy(epsilon, Twork);
    }
    std::string final_svd_strat = (svd_strat == "auto") ? auto_svd_strat : svd_strat;
    return std::make_tuple(epsilon, Twork, final_svd_strat);
}

// Function to canonicalize basis functions
inline void canonicalize(PiecewiseLegendrePolyVector &u,
                         PiecewiseLegendrePolyVector &v)
{
    for (size_t i = 0; i < u.size(); ++i) {
        double gauge = std::copysign(1.0, u.polyvec[i](1.0));
        u.polyvec[i].data *= gauge;
        v.polyvec[i].data *= gauge;
    }
}

// Base class for SVE strategies
template <typename K, typename T>
class AbstractSVE {
public:
    virtual ~AbstractSVE() { }
    virtual std::vector<Eigen::MatrixX<T>> matrices() const = 0;
    virtual SVEResult<K>
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const = 0;
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
    double epsilon;
    int n_gauss;
    int nsvals_hint;
    // Quadrature rules and segments
    Rule<T> rule;
    std::vector<T> segs_x;
    std::vector<T> segs_y;
    Rule<T> gauss_x;
    Rule<T> gauss_y;

    // Constructor
    SamplingSVE(K kernel_, double epsilon_, int n_gauss_ = -1)
        : kernel(kernel_), epsilon(epsilon_)
    {
        n_gauss =
            (n_gauss_ > 0) ? n_gauss_ : sve_hints(kernel, epsilon).ngauss();
        // TODO: Implement Rule<T>(n_gauss)
        rule = legendre<T>(n_gauss);
        auto hints = sve_hints(kernel, epsilon);
        nsvals_hint = hints.nsvals();
        segs_x = hints.template segments_x<T>();
        segs_y = hints.template segments_y<T>();
        gauss_x = rule.piecewise(segs_x);
        gauss_y = rule.piecewise(segs_y);
    }

    // Compute matrices for SVD
    std::vector<Eigen::MatrixX<T>> matrices() const override
    {
        std::vector<Eigen::MatrixX<T>> mats;
        Eigen::MatrixX<T> A = matrix_from_gauss(kernel, gauss_x, gauss_y);
        std::cout << "before scaling" << A(0, 0) << std::endl;
        // Element-wise multiplication with square roots of weights
        for (int i = 0; i < gauss_x.w.size(); ++i) {
            A.row(i) *= sqrt_impl(gauss_x.w[i]);
        }
        for (int j = 0; j < gauss_y.w.size(); ++j) {
            A.col(j) *= sqrt_impl(gauss_y.w[j]);
        }
        mats.push_back(A);
        return mats;
    }

    // Postprocess to construct SVEResult
    SVEResult<K>
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const override
    {
        // Assuming there's only one matrix in u_list, s_list, and v_list
        const Eigen::MatrixX<T> &u = u_list[0];
        const Eigen::VectorX<T> &s_ = s_list[0];
        const Eigen::MatrixX<T> &v = v_list[0];

        Eigen::VectorXd s = s_.template cast<double>();
        Eigen::VectorX<T> gauss_x_w =
            Eigen::VectorX<T>::Map(gauss_x.w.data(), gauss_x.w.size());
        Eigen::VectorX<T> gauss_y_w =
            Eigen::VectorX<T>::Map(gauss_y.w.data(), gauss_y.w.size());

        Eigen::MatrixX<T> u_x = u.array().colwise() / gauss_x_w.array().sqrt();
        Eigen::MatrixX<T> v_y = v.array().colwise() / gauss_y_w.array().sqrt();

        std::vector<T> u_x_flatten(u_x.data(), u_x.data() + u_x.size());
        std::vector<T> v_y_flatten(v_y.data(), v_y.data() + v_y.size());

        Eigen::TensorMap<Eigen::Tensor<T, 3>> u_data(
            u_x_flatten.data(), n_gauss, segs_x.size() - 1, s.size());
        Eigen::TensorMap<Eigen::Tensor<T, 3>> v_data(
            v_y_flatten.data(), n_gauss, segs_y.size() - 1, s.size());

        Eigen::MatrixX<T> cmat = legendre_collocation<T>(rule);

        for (int j = 0; j < u_data.dimension(1); ++j) {
            for (int k = 0; k < u_data.dimension(2); ++k) {
                for (int i = 0; i < cmat.rows(); ++i) {
                    u_data(i, j, k) = T(0);
                    for (int l = 0; l < cmat.cols(); ++l) {
                        u_data(i, j, k) += cmat(i, l) * u_data(l, j, k);
                    }
                }
            }
        }

        for (int j = 0; j < v_data.dimension(1); ++j) {
            for (int k = 0; k < v_data.dimension(2); ++k) {
                for (int i = 0; i < cmat.rows(); ++i) {
                    v_data(i, j, k) = T(0);
                    for (int l = 0; l < cmat.cols(); ++l) {
                        v_data(i, j, k) += cmat(i, l) * v_data(l, j, k);
                    }
                }
            }
        }
        // Manually compute differences for dsegs_x and dsegs_y
        Eigen::VectorX<T> dsegs_x(segs_x.size() - 1);
        for (int i = 0; i < segs_x.size() - 1; ++i) {
            dsegs_x[i] = segs_x[i + 1] - segs_x[i];
        }

        Eigen::VectorX<T> dsegs_y(segs_y.size() - 1);
        for (int i = 0; i < segs_y.size() - 1; ++i) {
            dsegs_y[i] = segs_y[i + 1] - segs_y[i];
        }

        // u_data_3d = u_data_3d * (T(0.5) *
        // dsegs_x).sqrt().transpose().matrix().asDiagonal(); v_data_3d =
        // v_data_3d * (T(0.5) *
        // dsegs_y).sqrt().transpose().matrix().asDiagonal();

        // Using nested for loops to multiply u_data
        for (int j = 0; j < u_data.dimension(1); ++j) {
            for (int k = 0; k < u_data.dimension(2); ++k) {
                for (int i = 0; i < u_data.dimension(0); ++i) {
                    u_data(i, j, k) *= sqrt_impl(T(0.5) * dsegs_x[j]);
                }
            }
        }

        // Using nested for loops to multiply v_data
        for (int j = 0; j < v_data.dimension(1); ++j) {
            for (int k = 0; k < v_data.dimension(2); ++k) {
                for (int i = 0; i < v_data.dimension(0); ++i) {
                    v_data(i, j, k) *= sqrt_impl(T(0.5) * dsegs_y[j]);
                }
            }
        }

        std::vector<PiecewiseLegendrePoly> polyvec_u;
        std::vector<PiecewiseLegendrePoly> polyvec_v;

        for (int i = 0; i < u_data.dimension(2); ++i) {
            Eigen::MatrixXd slice_double(u_data.dimension(0),
                                         u_data.dimension(1));
            for (int j = 0; j < u_data.dimension(0); ++j) {
                for (int k = 0; k < u_data.dimension(1); ++k) {
                    slice_double(j, k) = static_cast<double>(u_data(j, k, i));
                }
            }
            std::vector<double> segs_x_double;
            segs_x_double.reserve(segs_x.size());
            for (const auto &x : segs_x) {
                segs_x_double.push_back(static_cast<double>(x));
            }
            polyvec_u.push_back(PiecewiseLegendrePoly(
                slice_double,
                Eigen::VectorXd::Map(segs_x_double.data(),
                                     segs_x_double.size()),
                i));
        }


        // Repeat similar changes for v_data
        for (int i = 0; i < v_data.dimension(2); ++i) {
            Eigen::MatrixXd slice_double(v_data.dimension(0),
                                         v_data.dimension(1));
            for (int j = 0; j < v_data.dimension(0); ++j) {
                for (int k = 0; k < v_data.dimension(1); ++k) {
                    slice_double(j, k) = static_cast<double>(v_data(j, k, i));
                }
            }

            std::vector<double> segs_y_double;
            segs_y_double.reserve(segs_y.size());
            for (const auto &y : segs_y) {
                segs_y_double.push_back(static_cast<double>(y));
            }

            polyvec_v.push_back(PiecewiseLegendrePoly(
                slice_double,
                Eigen::VectorXd::Map(segs_y_double.data(),
                                     segs_y_double.size()),
                i));
        }




        PiecewiseLegendrePolyVector ulx(polyvec_u);
        PiecewiseLegendrePolyVector vly(polyvec_v);
        canonicalize(ulx, vly);
        return SVEResult<K>(ulx, s, vly, kernel, epsilon);
    }
};

// CentrosymmSVE class
template <typename K, typename T = double>
class CentrosymmSVE : public AbstractSVE<K, T> {
public:
    K kernel;
    double epsilon;
    SamplingSVE<K, T> even;
    SamplingSVE<K, T> odd;
    int nsvals_hint;

    CentrosymmSVE(const K &kernel_, double epsilon_, int n_gauss_ = -1)
        : kernel(kernel_),
          epsilon(epsilon_),
          even(static_cast<K &>(*get_symmetrized(kernel_, +1)), epsilon_,
               n_gauss_),
          odd(static_cast<K &>(*get_symmetrized(kernel_, -1)), epsilon_,
              n_gauss_)
    {
        auto evenk_ = get_symmetrized(kernel_, +1);
        auto oddk_ = get_symmetrized(kernel_, -1);
        SamplingSVE<K, T> even_(static_cast<K &>(*evenk_), epsilon_, n_gauss_);
        SamplingSVE<K, T> odd_(static_cast<K &>(*oddk_), epsilon_, n_gauss_);
        auto evenk_copy_ = K(static_cast<K &>(*evenk_));
        auto oddk_copy_ = K(static_cast<K &>(*oddk_));
        std::cout << typeid(even.kernel).name() << std::endl;
        std::cout << typeid(odd.kernel).name() << std::endl;
        std::cout << "even at 0.5, 0.5 " << (*evenk_)(0.5, 0.5) << std::endl;
        std::cout << "odd at 0.5, 0.5 " << (*oddk_)(0.5, 0.5) << std::endl;
        std::cout << "even at 0.5, 0.5 " << (evenk_copy_)(0.5, 0.5) << std::endl;
        std::cout << "odd at 0.5, 0.5 " << (oddk_copy_)(0.5, 0.5) << std::endl;
        std::cout << "even at 0.5, 0.5 " << even_.kernel(0.5, 0.5) << std::endl;
        std::cout << "odd at 0.5, 0.5 " << odd_.kernel(0.5, 0.5) << std::endl;
        nsvals_hint = std::max(even.nsvals_hint, odd.nsvals_hint);
    }

    std::vector<Eigen::MatrixX<T>> matrices() const override
    {
        auto mats_even = even.matrices();
        auto mats_odd = odd.matrices();
        std::cout << "matrices even: " << mats_even[0].sum() << std::endl;
        std::cout << "matrices odd: " << mats_odd[0].sum() << std::endl;
        return {mats_even[0], mats_odd[0]};
    }

    // Replace the vector merging code with Eigen operations
    SVEResult<K>
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const override
    {
        SVEResult<K> result_even =
            even.postprocess({u_list[0]}, {s_list[0]}, {v_list[0]});
        SVEResult<K> result_odd =
            odd.postprocess({u_list[1]}, {s_list[1]}, {v_list[1]});

        // Merge results using vectors instead of insert
        std::vector<PiecewiseLegendrePoly> u_merged;
        u_merged.reserve(result_even.u.size() + result_odd.u.size());
        u_merged.insert(u_merged.end(), result_even.u.begin(),
                        result_even.u.end());
        u_merged.insert(u_merged.end(), result_odd.u.begin(),
                        result_odd.u.end());

        // Concatenate singular values
        Eigen::VectorXd s_merged(result_even.s.size() + result_odd.s.size());
        s_merged << result_even.s, result_odd.s;

        // Merge v vectors
        std::vector<PiecewiseLegendrePoly> v_merged;
        v_merged.reserve(result_even.v.size() + result_odd.v.size());
        v_merged.insert(v_merged.end(), result_even.v.begin(),
                        result_even.v.end());
        v_merged.insert(v_merged.end(), result_odd.v.begin(),
                        result_odd.v.end());

        // For segments, use the hints from the kernel class
        auto hints = sve_hints(kernel, epsilon);
        auto segs_x_full = hints.template segments_x<T>();
        auto segs_y_full = hints.template segments_y<T>();

        // Rest of the implementation...
        // Create PiecewiseLegendrePolyVector from merged vectors
        PiecewiseLegendrePolyVector u_complete(u_merged);
        PiecewiseLegendrePolyVector v_complete(v_merged);

        return SVEResult<K>(u_complete, s_merged, v_complete, kernel, epsilon);
    }
};

// SVEResult class
template <typename K>
class SVEResult {
public:
    PiecewiseLegendrePolyVector u;
    Eigen::VectorXd s;
    PiecewiseLegendrePolyVector v;

    K kernel;
    double epsilon;
    // Default constructor
    SVEResult() {}
    // Constructor
    SVEResult(const PiecewiseLegendrePolyVector &u_, const Eigen::VectorXd &s_,
              const PiecewiseLegendrePolyVector &v_, const K &kernel_,
              double epsilon_)
        : u(u_), s(s_), v(v_), kernel(kernel_), epsilon(epsilon_)
    {
    }

    std::tuple<PiecewiseLegendrePolyVector, Eigen::VectorXd, PiecewiseLegendrePolyVector>
    part(double eps = std::numeric_limits<double>::quiet_NaN(),
         int max_size = -1) const
    {
        eps = std::isnan(eps) ? this->epsilon : eps;
        double threshold = eps * s(0);

        int cut =
            std::count_if(s.data(), s.data() + s.size(),
                          [threshold](double val) { return val >= threshold; });

        if (max_size > 0) {
            cut = std::min(cut, max_size);
        }
        std::vector<PiecewiseLegendrePoly> u_part_(u.begin(), u.begin() + cut);
        PiecewiseLegendrePolyVector u_part(u_part_);
        Eigen::VectorXd s_part(s.head(cut));
        std::vector<PiecewiseLegendrePoly> v_part_(v.begin(), v.begin() + cut);
        PiecewiseLegendrePolyVector v_part(v_part_);
        return std::make_tuple(u_part, s_part, v_part);
    }
};

template <typename K, typename T>
std::shared_ptr<AbstractSVE<K, T>>
determine_sve(const K &kernel, double safe_epsilon, int n_gauss)
{
    if (kernel.is_centrosymmetric()) {
        std::cout << "Centrosymmetric kernel detected." << std::endl;
        return std::make_shared<CentrosymmSVE<K, T>>(kernel, safe_epsilon,
                                                     n_gauss);
    } else {
        std::cout << "Non-centrosymmetric kernel detected." << std::endl;
        return std::make_shared<SamplingSVE<K, T>>(kernel, safe_epsilon,
                                                   n_gauss);
    }
}

// Function to truncate singular values
template <typename T>
inline std::tuple<std::vector<Eigen::MatrixX<T>>,
                  std::vector<Eigen::VectorX<T>>,
                  std::vector<Eigen::MatrixX<T>>>
truncate(std::vector<Eigen::MatrixX<T>> &u_list,
         std::vector<Eigen::VectorX<T>> &s_list,
         std::vector<Eigen::MatrixX<T>> &v_list, T rtol = 0.0,
         int lmax = std::numeric_limits<int>::max())
{
    std::vector<Eigen::MatrixX<T>> u_list_truncated;
    std::vector<Eigen::VectorX<T>> s_list_truncated;
    std::vector<Eigen::MatrixX<T>> v_list_truncated;
    // Collect all singular values
    std::vector<T> all_singular_values;
    for (const auto &s : s_list) {
        for (int i = 0; i < s.size(); ++i) {
            all_singular_values.push_back(s(i));
        }
    }
    std::sort(all_singular_values.begin(), all_singular_values.end(),
              std::greater<T>());

    // Determine cutoff
    T cutoff = rtol * all_singular_values.front();
    if (lmax < static_cast<int>(all_singular_values.size())) {
        cutoff = std::max(cutoff, all_singular_values[lmax - 1]);
    }

    // Truncate singular values and corresponding vectors
    for (size_t idx = 0; idx < s_list.size(); ++idx) {
        const auto &s = s_list[idx];
        int scount = 0;
        for (int i = 0; i < s.size(); ++i) {
            if (s(i) > cutoff) {
                ++scount;
            } else {
                break;
            }
        }
        if (scount < s.size()) {
            u_list_truncated.push_back(u_list[idx].leftCols(scount));
            s_list_truncated.push_back(s_list[idx].head(scount));
            v_list_truncated.push_back(v_list[idx].leftCols(scount));
        }
    }
    return std::make_tuple(u_list_truncated, s_list_truncated,
                           v_list_truncated);
}

template <typename K, typename T>
auto pre_postprocess(K &kernel, double safe_epsilon, int n_gauss,
                     double cutoff = std::numeric_limits<double>::quiet_NaN(),
                     int lmax = -1)
{
    auto sve = determine_sve<K, T>(kernel, safe_epsilon, n_gauss);
    // Compute SVDs
    std::cout << "Computing SVDs..." << std::endl;
    std::vector<Eigen::MatrixX<T>> matrices = sve->matrices();
    // TODO: implement SVD Resutls
    std::vector<
        std::tuple<Eigen::MatrixX<T>, Eigen::MatrixX<T>, Eigen::MatrixX<T>>>
        svds;
    for (const auto &mat : matrices) {
        std::cout << mat(0, 0) << std::endl;
        std::cout << mat(1, 0) << std::endl;
        auto svd = sparseir::compute_svd(mat);
        svds.push_back(svd);
    }

    // Extract singular values and vectors
    std::vector<Eigen::MatrixX<T>> u_list_, v_list_;
    std::vector<Eigen::VectorX<T>> s_list_;
    for (const auto &svd : svds) {
        auto u = std::get<0>(svd);
        auto s = std::get<1>(svd);
        auto v = std::get<2>(svd);
        u_list_.push_back(u);
        s_list_.push_back(s);
        v_list_.push_back(v);
    }

    // Apply cutoff and lmax
    T cutoff_actual = std::isnan(cutoff)
                          ? 2 * T(std::numeric_limits<double>::epsilon())
                          : T(cutoff);
    std::vector<Eigen::MatrixX<T>> u_list_truncated;
    std::vector<Eigen::VectorX<T>> s_list_truncated;
    std::vector<Eigen::MatrixX<T>> v_list_truncated;
    std::tie(u_list_truncated, s_list_truncated, v_list_truncated) =
        truncate(u_list_, s_list_, v_list_, cutoff_actual, lmax);
    // Postprocess to get the SVEResult
    return sve->postprocess(u_list_truncated, s_list_truncated,
                            v_list_truncated);
}

// Function to compute SVE result
template <typename K>
SVEResult<K> compute_sve(K kernel, double epsilon = std::numeric_limits<double>::quiet_NaN(),
            double cutoff = std::numeric_limits<double>::quiet_NaN(),
            std::string Twork = "",
            int lmax = std::numeric_limits<int>::max(), int n_gauss = -1,
            const std::string &svd_strat = "auto")
{
    // Choose accuracy parameters
    double safe_epsilon;
    std::string Twork_actual;
    std::string svd_strategy_actual;
    std::cout << "Twork: " << Twork << std::endl;
    std::cout << "svd_strat: " << svd_strat << std::endl;
    std::tie(safe_epsilon, Twork_actual, svd_strategy_actual) =
        choose_accuracy(epsilon, Twork, svd_strat);
    //std::cout << "Twork_actual: " << Twork_actual << std::endl;
    //std::cout << "svd_strategy_actual: " << svd_strategy_actual << std::endl;

    if (Twork_actual == "Float64") {
        return pre_postprocess<K, double>(kernel, safe_epsilon, n_gauss, cutoff,
                                          lmax);
    } else {
        // xprec::DDouble
        return pre_postprocess<K, xprec::DDouble>(kernel, safe_epsilon, n_gauss,
                                                  cutoff, lmax);
    }
}

} // namespace sparseir
