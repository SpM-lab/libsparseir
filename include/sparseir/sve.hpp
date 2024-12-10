// libsparseir/include/sparseir/sve.hpp

#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor> // Include Tensor module
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

inline std::tuple<double, std::string, std::string> choose_accuracy(double epsilon, std::string Twork) {
    if (Twork == "Float64") {
        if (epsilon >= std::sqrt(std::numeric_limits<double>::epsilon())) {
            return std::make_tuple(epsilon, Twork, "default");
        } else {
            std::cerr << "Warning: Basis cutoff is " << epsilon << ", which is below √ε with ε = "
                    << std::numeric_limits<double>::epsilon() << ".\n"
                    << "Expect singular values and basis functions for large l to have lower precision than the cutoff.\n";
            return std::make_tuple(epsilon, Twork, "accurate");
        }
    } else
    {
        // Handle the case for xprec::DDouble
        if (epsilon >= std::sqrt(std::numeric_limits<double>::epsilon()))
        {
            return std::make_tuple(epsilon, Twork, "default");
        }
        else
        {
            std::cerr << "Warning: Basis cutoff is " << epsilon << ", which is below √ε with ε = "
                      << std::numeric_limits<xprec::DDouble>::epsilon() << ".\n"
                      << "Expect singular values and basis functions for large l to have lower precision than the cutoff.\n";
            return std::make_tuple(epsilon, Twork, "accurate");
        }
    }

}

inline std::tuple<double, std::string, std::string> choose_accuracy(double epsilon, std::nullptr_t) {
    if (epsilon >= std::sqrt(std::numeric_limits<double>::epsilon())) {
        return std::make_tuple(epsilon, "Float64", "default");
    } else {
        /*
        // This should work, but catch2 can't catch this warning
        // Therefore we suppress this block
        if (epsilon < std::sqrt(std::numeric_limits<double>::epsilon())) {
            std::cerr << "Warning: Basis cutoff is " << epsilon << ", which is below √ε with ε = "
                      << std::numeric_limits<double>::epsilon() << ".\n"
                      << "Expect singular values and basis functions for large l to have lower precision than the cutoff.\n";
        }
        */
        return std::make_tuple(epsilon, "Float64x2", "default");
    }
}

// Equivalent to Julia implementation:
// julia> choose_accuracy(::Nothing, Twork) = sqrt(eps(Twork)), Twork, :default
inline std::tuple<double, std::string, std::string> choose_accuracy(std::nullptr_t, std::string Twork) {
    if (Twork == "Float64x2"){
        const double epsilon = 2.220446049250313e-16; // julia> using MultiFloats; Float64(sqrt(eps(Float64x2)))
        return std::make_tuple(epsilon, Twork, "default");
    } else {
        return std::make_tuple(std::sqrt(std::numeric_limits<double>::epsilon()), Twork, "default");
    }
}

// Equivalent to Julia implementation:
// julia> choose_accuracy(::Nothing, ::Nothing) = Float64(sqrt(eps(T_MAX))), T_MAX, :default
inline std::tuple<double, std::string, std::string> choose_accuracy(std::nullptr_t, std::nullptr_t) {
    const double epsilon = 2.220446049250313e-16; // julia> using MultiFloats; Float64(sqrt(eps(Float64x2)))
    return std::make_tuple(epsilon, "Float64x2", "default");
}

inline std::tuple<double, std::string, std::string> choose_accuracy(double epsilon, std::string Twork, std::string svd_strat) {
    std::string auto_svd_strat;
    std::tie(epsilon, Twork, auto_svd_strat) = choose_accuracy(epsilon, Twork);
    std::string final_svd_strat = (svd_strat == "auto") ? auto_svd_strat : svd_strat;
    return std::make_tuple(epsilon, Twork, final_svd_strat);
}

// Function to canonicalize basis functions
inline void canonicalize(
    PiecewiseLegendrePolyVector &u,
    PiecewiseLegendrePolyVector &v)
{
    for (size_t i = 0; i < u.size(); ++i)
    {
        double gauge = std::copysign(1.0, u.polyvec[i](1.0));
        u.polyvec[i].data *= gauge;
        v.polyvec[i].data *= gauge;
    }
}

// Base class for SVE strategies
template <typename K, typename T>
class AbstractSVE {
public:
    virtual ~AbstractSVE() {}
    virtual std::vector<Eigen::MatrixX<T>> matrices() const = 0;
    virtual SVEResult<K> postprocess(
        const std::vector<Eigen::MatrixX<T>>& u_list,
        const std::vector<Eigen::VectorX<T>>& s_list,
        const std::vector<Eigen::MatrixX<T>>& v_list
    ) const = 0;
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
        : kernel(kernel_),
          epsilon(epsilon_)
    {
        n_gauss = (n_gauss_ > 0) ? n_gauss_ : sve_hints(kernel, epsilon).ngauss();
        // TODO: Implement Rule<T>(n_gauss)
        rule = convert<T>(legendre(n_gauss));
        auto hints = sve_hints(kernel, epsilon);
        nsvals_hint = hints.nsvals();
        segs_x = hints.template segments_x<T>();
        segs_y = hints.template segments_y<T>();
        gauss_x = rule.piecewise(segs_x);
        gauss_y = rule.piecewise(segs_y);
    }

    // Compute matrices for SVD
    std::vector<Eigen::MatrixX<T>> matrices() const override {
        std::vector<Eigen::MatrixX<T>> mats;
        Eigen::MatrixX<T> A = matrix_from_gauss(kernel, gauss_x, gauss_y);
        // Element-wise multiplication with square roots of weights
        for (int i = 0; i < gauss_x.w.size(); ++i)
        {
            A.row(i) *= std::sqrt(gauss_x.w[i]);
        }
        for (int j = 0; j < gauss_y.w.size(); ++j)
        {
            A.col(j) *= std::sqrt(gauss_y.w[j]);
        }
        mats.push_back(A);
        return mats;
    }

    // Postprocess to construct SVEResult
    SVEResult<K> postprocess(
        const std::vector<Eigen::MatrixX<T>> &u_list,
        const std::vector<Eigen::VectorX<T>> &s_list,
        const std::vector<Eigen::MatrixX<T>> &v_list) const override
    {
        // Assuming there's only one matrix in u_list, s_list, and v_list
        const Eigen::MatrixX<T> &u = u_list[0];
        const Eigen::VectorX<T> &s_ = s_list[0];
        const Eigen::MatrixX<T> &v = v_list[0];
        Eigen::VectorX<T> s = s_.template cast<double>();

        Eigen::VectorX<T> gauss_x_w = Eigen::VectorX<T>::Map(gauss_x.w.data(), gauss_x.w.size());
        Eigen::VectorX<T> gauss_y_w = Eigen::VectorX<T>::Map(gauss_y.w.data(), gauss_y.w.size());

        Eigen::MatrixX<T> u_x = u.array().colwise() / gauss_x_w.array().sqrt();
        Eigen::MatrixX<T> v_y = v.array().colwise() / gauss_y_w.array().sqrt();

        std::vector<T> u_x_flatten(u_x.data(), u_x.data() + u_x.size());
        std::vector<T> v_y_flatten(v_y.data(), v_y.data() + v_y.size());

        Eigen::TensorMap<Eigen::Tensor<T, 3>> u_data(u_x_flatten.data(), n_gauss, segs_x.size() - 1, s.size());
        Eigen::TensorMap<Eigen::Tensor<T, 3>> v_data(v_y_flatten.data(), n_gauss, segs_y.size() - 1, s.size());

        Eigen::MatrixX<T> cmat = legendre_collocation(rule);

        for (int j = 0; j < u_data.dimension(1); ++j)
        {
            for (int k = 0; k < u_data.dimension(2); ++k)
            {
                for (int i = 0; i < cmat.rows(); ++i)
                {
                    u_data(i, j, k) = T(0);
                    for (int l = 0; l < cmat.cols(); ++l)
                    {
                        u_data(i, j, k) += cmat(i, l) * u_data(l, j, k);
                    }
                }
            }
        }

        for (int j = 0; j < v_data.dimension(1); ++j)
        {
            for (int k = 0; k < v_data.dimension(2); ++k)
            {
                for (int i = 0; i < cmat.rows(); ++i)
                {
                    v_data(i, j, k) = T(0);
                    for (int l = 0; l < cmat.cols(); ++l)
                    {
                        v_data(i, j, k) += cmat(i, l) * v_data(l, j, k);
                    }
                }
            }
        }

        // Manually compute differences for dsegs_x and dsegs_y
        Eigen::VectorX<T> dsegs_x(segs_x.size() - 1);
        for (int i = 0; i < segs_x.size() - 1; ++i)
        {
            dsegs_x[i] = segs_x[i + 1] - segs_x[i];
        }

        Eigen::VectorX<T> dsegs_y(segs_y.size() - 1);
        for (int i = 0; i < segs_y.size() - 1; ++i)
        {
            dsegs_y[i] = segs_y[i + 1] - segs_y[i];
        }

        //u_data_3d = u_data_3d * (T(0.5) * dsegs_x).sqrt().transpose().matrix().asDiagonal();
        //v_data_3d = v_data_3d * (T(0.5) * dsegs_y).sqrt().transpose().matrix().asDiagonal();

        // Using nested for loops to multiply u_data
        for (int j = 0; j < u_data.dimension(1); ++j)
        {
            for (int k = 0; k < u_data.dimension(2); ++k)
            {
                for (int i = 0; i < u_data.dimension(0); ++i)
                {
                    u_data(i, j, k) *= std::sqrt(0.5 * dsegs_x[j]);
                }
            }
        }

        // Using nested for loops to multiply v_data
        for (int j = 0; j < v_data.dimension(1); ++j)
        {
            for (int k = 0; k < v_data.dimension(2); ++k)
            {
                for (int i = 0; i < v_data.dimension(0); ++i)
                {
                    v_data(i, j, k) *= std::sqrt(0.5 * dsegs_y[j]);
                }
            }
        }

        std::vector<PiecewiseLegendrePoly> polyvec_u;
        std::vector<PiecewiseLegendrePoly> polyvec_v;

        for (int i = 0; i < u_data.dimension(2); ++i)
        {
            Eigen::MatrixXd slice_double(u_data.dimension(0), u_data.dimension(1));
            for (int j = 0; j < u_data.dimension(1); ++j)
            {
                for (int k = 0; k < u_data.dimension(2); ++k)
                {
                    slice_double(j, k) = u_data(j, k, i);
                }
            }

            polyvec_u.push_back(PiecewiseLegendrePoly(slice_double, Eigen::VectorXd::Map(segs_x.data(), segs_x.size()), i));
        }

        // Repeat similar changes for v_data
        for (int i = 0; i < v_data.dimension(2); ++i)
        {
            Eigen::MatrixXd slice_double(v_data.dimension(0), v_data.dimension(1));
            for (int j = 0; j < v_data.dimension(1); ++j)
            {
                for (int k = 0; k < v_data.dimension(2); ++k)
                {
                    slice_double(j, k) = v_data(j, k, i);
                }
            }

            polyvec_v.push_back(PiecewiseLegendrePoly(slice_double, Eigen::VectorXd::Map(segs_y.data(), segs_y.size()), i));
        }
        PiecewiseLegendrePolyVector ulx(polyvec_u);
        PiecewiseLegendrePolyVector vly(polyvec_v);
        canonicalize(ulx, vly);
        return SVEResult<K>(ulx, s, vly, kernel, epsilon);
    }
};

// CentrosymmSVE class
template <typename K, typename T>
class CentrosymmSVE : public AbstractSVE<K, T> {
public:
    K kernel;
    double epsilon;
    SamplingSVE<K, T> even;
    SamplingSVE<K, T> odd;
    int nsvals_hint;

    CentrosymmSVE(const K& kernel_, double epsilon_, int n_gauss_ = -1)
        : kernel(kernel_),
          epsilon(epsilon_),
          // n_gauss(n_gauss_),
          even(get_symmetrized(kernel_, +1), epsilon_, n_gauss_),
          odd(get_symmetrized(kernel_, -1), epsilon_, n_gauss_) {
        nsvals_hint = std::max(even.nsvals_hint, odd.nsvals_hint);
    }

    std::vector<Eigen::MatrixX<T>> matrices() const override {
        Eigen::MatrixX<T> mats_even = even.matrices();
        Eigen::MatrixX<T> mats_odd = odd.matrices();
        return std::vector<Eigen::MatrixX<T>>{ mats_even[0], mats_odd[0] };
    }

    SVEResult<K> postprocess(const std::vector<Eigen::MatrixX<T>>& u_list,
                                   const std::vector<Eigen::VectorX<T>>& s_list,
                                   const std::vector<Eigen::MatrixX<T>>& v_list) const override {
        SVEResult<K> result_even = even.postprocess({ u_list[0] }, { s_list[0] }, { v_list[0] });
        SVEResult<K> result_odd = odd.postprocess({ u_list[1] }, { s_list[1] }, { v_list[1] });


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
    Eigen::VectorXd s;
    PiecewiseLegendrePolyVector v;

    K kernel;
    double epsilon;

    SVEResult(const PiecewiseLegendrePolyVector& u_, const Eigen::VectorXd& s_,
              const PiecewiseLegendrePolyVector& v_, const K& kernel_, double epsilon_)
        : u(u_), s(s_), v(v_), kernel(kernel_), epsilon(epsilon_) {}
};

template <typename K, typename T>
std::shared_ptr<AbstractSVE<K, T>> determine_sve(const K& kernel, double safe_epsilon, int n_gauss) {
    if (kernel.is_centrosymmetric()) {
        return std::make_shared<CentrosymmSVE<K, T>>(kernel, safe_epsilon, n_gauss);
    } else {
        return std::make_shared<SamplingSVE<K, T>>(kernel, safe_epsilon, n_gauss);
    }
}

// Function to truncate singular values
inline void truncate_singular_values(
    std::vector<Eigen::MatrixXd> &u_list,
    std::vector<Eigen::VectorXd> &s_list,
    std::vector<Eigen::MatrixXd> &v_list,
    double rtol,
    int lmax)
{
    // Collect all singular values
    std::vector<double> all_singular_values;
    for (const auto &s : s_list)
    {
        for (int i = 0; i < s.size(); ++i)
        {
            all_singular_values.push_back(s(i));
        }
    }
    std::sort(all_singular_values.begin(), all_singular_values.end(), std::greater<double>());

    // Determine cutoff
    double cutoff = rtol * all_singular_values.front();
    if (lmax < static_cast<int>(all_singular_values.size()))
    {
        cutoff = std::max(cutoff, all_singular_values[lmax - 1]);
    }

    // Truncate singular values and corresponding vectors
    for (size_t idx = 0; idx < s_list.size(); ++idx)
    {
        const auto &s = s_list[idx];
        int scount = 0;
        for (int i = 0; i < s.size(); ++i)
        {
            if (s(i) > cutoff)
            {
                ++scount;
            }
            else
            {
                break;
            }
        }
        if (scount < s.size())
        {
            u_list[idx] = u_list[idx].leftCols(scount);
            s_list[idx] = s_list[idx].head(scount);
            v_list[idx] = v_list[idx].leftCols(scount);
        }
    }
}


template <typename K, typename T>
auto pre_postprocess(K &kernel, double safe_epsilon, int n_gauss, double cutoff = std::numeric_limits<double>::quiet_NaN(), int lmax = -1)
{
    auto sve = determine_sve<K, T>(kernel, safe_epsilon, n_gauss);
    // Compute SVDs
    std::vector<Eigen::MatrixX<T>> matrices = sve.matrices();
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
    T cutoff_actual = std::isnan(cutoff) ? 2 * std::numeric_limits<T>::epsilon() : cutoff;
    //truncate_singular_values(u_list, s_list, v_list, cutoff_actual, lmax);
    // Postprocess to get the SVEResult
    return sve.postprocess(u_list, s_list, v_list);
}

// Function to compute SVE result
template <typename K>
    auto compute_sve(K kernel,
                            std::string Twork = "Floatt64",
                            double cutoff = std::numeric_limits<double>::quiet_NaN(),
                            double epsilon = std::numeric_limits<double>::quiet_NaN(),
                            int lmax = std::numeric_limits<int>::max(),
                            int n_gauss = -1,
                            const std::string& svd_strat = "auto") {
    // Choose accuracy parameters
    double safe_epsilon;
    std::string Twork_actual;
    std::string svd_strategy_actual;
    std::cout << "Twork: " << Twork << std::endl;
    std::cout << "svd_strat: " << svd_strat << std::endl;
    std::tie(safe_epsilon, Twork_actual, svd_strategy_actual) = choose_accuracy(epsilon, Twork, svd_strat);
    std::cout << "Twork_actual: " << Twork_actual << std::endl;
    std::cout << "svd_strategy_actual: " << svd_strategy_actual << std::endl;
    if (Twork_actual == "Float64"){
        return pre_postprocess<K, double>(kernel, safe_epsilon, n_gauss, cutoff, lmax);
    }
    else{
        // xprec::DDouble
        return pre_postprocess<K, xprec::DDouble>(kernel, safe_epsilon, n_gauss, cutoff, lmax);
    }
}

} // namespace sparseir
