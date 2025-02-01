// libsparseir/include/sparseir/sve.hpp

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
#include <unsupported/Eigen/CXX11/Tensor> // Include Tensor module
#include <vector>

// Include other necessary headers here

namespace sparseir {


// SVEResult class
class SVEResult {
public:
    PiecewiseLegendrePolyVector u;
    Eigen::VectorXd s;
    PiecewiseLegendrePolyVector v;

    double epsilon;
    // Default constructor
    SVEResult() {}
    // Constructor
    SVEResult(const PiecewiseLegendrePolyVector &u_, const Eigen::VectorXd &s_,
              const PiecewiseLegendrePolyVector &v_,
              double epsilon_)
        : u(u_), s(s_), v(v_), epsilon(epsilon_)
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


inline std::tuple<double, std::string, std::string>
choose_accuracy(double epsilon, const std::string &Twork)
{
    if (Twork != "Float64" && Twork != "Float64x2") {
        throw std::invalid_argument("Twork must be either 'Float64' or 'Float64x2'");
    }
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
auto_choose_accuracy(double epsilon, std::string Twork, std::string svd_strat = "auto")
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

// SamplingSVE class
/**
 * @brief Sampling-based SVE to SVD translation.
 *
 * Maps the singular value expansion (SVE) of a kernel onto the singular
 * value decomposition (SVD) of a matrix by choosing two sets of Gauss
 * quadrature rules and approximating the integrals in the SVE equations
 * by finite sums.
 */
template <typename K, typename T>
class SamplingSVE : public AbstractSVE<T> {
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
    SamplingSVE(const K &kernel_, double epsilon_, int n_gauss_ = -1)
        : kernel(kernel_), epsilon(epsilon_)
    {
        auto hints = kernel.template sve_hints<T>(epsilon);
        n_gauss =
            (n_gauss_ > 0) ? n_gauss_ : hints->ngauss();
        // TODO: Implement Rule<T>(n_gauss)
        rule = legendre<T>(n_gauss);
        nsvals_hint = hints->nsvals();
        segs_x = hints->segments_x();
        segs_y = hints->segments_y();
        gauss_x = rule.piecewise(segs_x);
        gauss_y = rule.piecewise(segs_y);
    }

    // Compute matrices for SVD
    std::vector<Eigen::MatrixX<T>> matrices() const override
    {
        std::vector<Eigen::MatrixX<T>> mats;
        Eigen::MatrixX<T> A = matrix_from_gauss(kernel, gauss_x, gauss_y);
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
    SVEResult
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
        return SVEResult(ulx, s, vly, epsilon);
    }
};

// CentrosymmSVE class
template <typename K, typename T>
class CentrosymmSVE : public AbstractSVE<T> {
public:
    K kernel;
    double epsilon;
    SamplingSVE<typename SymmKernelTraits<K, std::integral_constant<int, +1>>::type, T> even;
    SamplingSVE<typename SymmKernelTraits<K, std::integral_constant<int, -1>>::type, T> odd;
    int nsvals_hint;

    CentrosymmSVE(const K &kernel_, double epsilon_, int n_gauss_ = -1)
        : kernel(kernel_),
          epsilon(epsilon_),
          even(get_symmetrized(kernel_, std::integral_constant<int, +1>{}), epsilon_, n_gauss_),
          odd(get_symmetrized(kernel_, std::integral_constant<int, -1>{}), epsilon_, n_gauss_)
    {
        /*
        auto evenk_ = get_symmetrized(kernel_, +1);
        auto oddk_ = get_symmetrized(kernel_, -1);
        SamplingSVE<K, T> even_(static_cast<K &>(*evenk_), epsilon_, n_gauss_);
        SamplingSVE<K, T> odd_(static_cast<K &>(*oddk_), epsilon_, n_gauss_);
        auto evenk_copy_ = K(static_cast<K &>(*evenk_));
        auto oddk_copy_ = K(static_cast<K &>(*oddk_));
        //std::cout << typeid(even.kernel).name() << std::endl;
        std::cout << typeid(odd.kernel).name() << std::endl;
        std::cout << "even at 0.5, 0.5 " << (*evenk_)(0.5, 0.5) << std::endl;
        std::cout << "odd at 0.5, 0.5 " << (*oddk_)(0.5, 0.5) << std::endl;
        std::cout << "even at 0.5, 0.5 " << (evenk_copy_)(0.5, 0.5) << std::endl;
        std::cout << "odd at 0.5, 0.5 " << (oddk_copy_)(0.5, 0.5) << std::endl;
        std::cout << "even at 0.5, 0.5 " << even_.kernel(0.5, 0.5) << std::endl;
        std::cout << "odd at 0.5, 0.5 " << odd_.kernel(0.5, 0.5) << std::endl;
        */
        nsvals_hint = std::max(even.nsvals_hint, odd.nsvals_hint);
    }

    std::vector<Eigen::MatrixX<T>> matrices() const override
    {
        auto mats_even = even.matrices();
        auto mats_odd = odd.matrices();
        return {mats_even[0], mats_odd[0]};
    }

    // Replace the vector merging code with Eigen operations
    SVEResult
    postprocess(const std::vector<Eigen::MatrixX<T>> &u_list,
                const std::vector<Eigen::VectorX<T>> &s_list,
                const std::vector<Eigen::MatrixX<T>> &v_list) const override
    {
        SVEResult result_even =
            even.postprocess({u_list[0]}, {s_list[0]}, {v_list[0]});
        SVEResult result_odd =
            odd.postprocess({u_list[1]}, {s_list[1]}, {v_list[1]});

        // Merge results using vectors instead of insert
        std::vector<PiecewiseLegendrePoly> u_merged;
        u_merged.reserve(result_even.u.size() + result_odd.u.size());
        u_merged.insert(u_merged.end(), result_even.u.begin(),
                        result_even.u.end());
        u_merged.insert(u_merged.end(), result_odd.u.begin(),
                        result_odd.u.end());

        // Concatenate singular values
        // TODO: sort singular values
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
        auto hints = sve_hints<T>(kernel, epsilon);
        auto segs_x_full = hints.segments_x();
        auto segs_y_full = hints.segments_y();

        // Rest of the implementation...
        // Create PiecewiseLegendrePolyVector from merged vectors
        PiecewiseLegendrePolyVector _u_complete(u_merged);
        PiecewiseLegendrePolyVector _v_complete(v_merged);

        // julia> signs = [fill(1, length(s_even)); fill(-1, length(s_odd))]
        Eigen::VectorXi sign_even = Eigen::VectorXi::Ones(result_even.s.size());
        Eigen::VectorXi sign_odd = -Eigen::VectorXi::Ones(result_odd.s.size());
        Eigen::VectorXi signs = Eigen::VectorXi::Zero(s_merged.size());
        signs << sign_even, sign_odd;

        /*
            Sort: now for totally positive kernels like defined in this
            module, this strictly speaking is not necessary as we know that the
            even/odd functions intersperse.
        */
        // julia> sort = sortperm(s; rev=true)
        // Get the sorted permutation indices
        std::vector<size_t> sorted_indices = sortperm_rev(s_merged);

        assert(sorted_indices.size() == s_merged.size());

        // Apply the sorted permutation to u_complete, v_complete, signs, and s_merged
        std::vector<PiecewiseLegendrePoly> u_sorted(sorted_indices.size());
        std::vector<PiecewiseLegendrePoly> v_sorted(sorted_indices.size());
        Eigen::VectorX<T> signs_sorted(sorted_indices.size());
        Eigen::VectorXd s_sorted(sorted_indices.size());

        for (size_t i = 0; i < sorted_indices.size(); ++i) {
            u_sorted[i] = u_merged[sorted_indices[i]];
            v_sorted[i] = v_merged[sorted_indices[i]];
            s_sorted[i] = s_merged[sorted_indices[i]];
            signs_sorted[i] = signs[sorted_indices[i]];
        }

        auto full_hints = sve_hints<T>(kernel, epsilon);
        auto segs_x_vec = full_hints.segments_x();
        auto segs_y_vec = full_hints.segments_y();

        std::vector<double> segs_x_double(segs_x_vec.size());
        std::vector<double> segs_y_double(segs_y_vec.size());

        // Convert xprec::DDouble to double
        for (size_t i = 0; i < segs_x_vec.size(); ++i) {
            segs_x_double[i] = static_cast<double>(segs_x_vec[i]);
        }
        for (size_t i = 0; i < segs_y_vec.size(); ++i) {
            segs_y_double[i] = static_cast<double>(segs_y_vec[i]);
        }
        // Use the double vectors instead
        Eigen::VectorXd segs_x = Eigen::Map<Eigen::VectorXd>(
            segs_x_double.data(), segs_x_double.size());
        Eigen::VectorXd segs_y = Eigen::Map<Eigen::VectorXd>(
            segs_y_double.data(), segs_y_double.size());

        std::vector<PiecewiseLegendrePoly> u_complete_vec;
        std::vector<PiecewiseLegendrePoly> v_complete_vec;

        Eigen::VectorXd poly_flip_x(u_sorted[0].data.rows());
        for (int i = 0; i < u_sorted[0].data.rows(); ++i) {
            poly_flip_x(i) = (i % 2 == 0) ? 1.0 : -1.0;
        }

        for (size_t i = 0; i < u_sorted.size(); ++i) {
            // Convert the data to double precision
            Eigen::MatrixXd u_pos_data = u_sorted[i].data.template cast<double>() / std::sqrt(2);
            Eigen::MatrixXd v_pos_data = v_sorted[i].data.template cast<double>() / std::sqrt(2);

            Eigen::MatrixXd u_neg_data = u_pos_data.rowwise().reverse();
            u_neg_data = u_neg_data.array().colwise() * (poly_flip_x * signs[i]).array();
            Eigen::MatrixXd v_neg_data = v_pos_data.rowwise().reverse();
            v_neg_data = v_neg_data.array().colwise() * (poly_flip_x * signs[i]).array();

            /*
            julia> u_data = hcat(u_neg_data, u_pos_data)
            julia> v_data = hcat(v_neg_data, v_pos_data)
            */
            Eigen::MatrixXd u_data = Eigen::MatrixXd::Zero(u_pos_data.rows(), u_neg_data.cols() + u_pos_data.cols());
            u_data.leftCols(u_neg_data.cols()) = u_neg_data;
            u_data.rightCols(u_pos_data.cols()) = u_pos_data;
            Eigen::MatrixXd v_data = Eigen::MatrixXd::Zero(v_pos_data.rows(), v_neg_data.cols() + v_pos_data.cols());
            v_data.leftCols(v_neg_data.cols()) = v_neg_data;
            v_data.rightCols(v_pos_data.cols()) = v_pos_data;

            Eigen::VectorXd segs_x_diff = segs_x.tail(segs_x.size() - 1) - segs_x.head(segs_x.size() - 1);
            Eigen::VectorXd segs_y_diff = segs_y.tail(segs_y.size() - 1) - segs_y.head(segs_y.size() - 1);

            // Create and store the polynomials
            u_complete_vec.push_back(PiecewiseLegendrePoly(u_data, segs_x, i, segs_x_diff, signs[i]));
            v_complete_vec.push_back(PiecewiseLegendrePoly(v_data, segs_y, i, segs_y_diff, signs[i]));
        }

        // Create the final vectors
        PiecewiseLegendrePolyVector u_complete(u_complete_vec);
        PiecewiseLegendrePolyVector v_complete(v_complete_vec);

        return SVEResult(u_complete, s_sorted, v_complete, epsilon);
    }
};

template <typename K, typename T>
std::shared_ptr<AbstractSVE<T>>
determine_sve(const K &kernel, double safe_epsilon, int n_gauss)
{
    if (kernel.is_centrosymmetric()) {
        return std::make_shared<CentrosymmSVE<K, T>>(kernel, safe_epsilon,
                                                     n_gauss);
    } else {
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
auto pre_postprocess(const K &kernel, double safe_epsilon, int n_gauss,
                     double cutoff = std::numeric_limits<double>::quiet_NaN(),
                     int lmax = -1)
{
    auto sve = determine_sve<K, T>(kernel, safe_epsilon, n_gauss);
    // Compute SVDs
    //std::cout << "Computing SVDs..." << std::endl;
    std::vector<Eigen::MatrixX<T>> matrices = sve->matrices();
    // TODO: implement SVD Resutls
    std::vector<
        std::tuple<Eigen::MatrixX<T>, Eigen::MatrixX<T>, Eigen::MatrixX<T>>>
        svds;
    for (const auto &mat : matrices) {
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
template <typename K >
SVEResult compute_sve(const K &kernel, double epsilon,
            double cutoff = std::numeric_limits<double>::quiet_NaN(),
            int lmax = std::numeric_limits<int>::max(),
            int n_gauss = -1,
            std::string Twork = "Float64x2"
            )
{
    // TODO: Sort out the logic
    double safe_epsilon;
    std::string Twork_actual;
    std::string svd_strategy_actual;
    std::tie(safe_epsilon, Twork_actual, svd_strategy_actual) = sparseir::auto_choose_accuracy(epsilon, Twork);

    if (Twork == "Float64") {
        return pre_postprocess<K, double>(kernel, safe_epsilon, n_gauss, cutoff, lmax);
    } else if (Twork == "Float64x2") {
        return pre_postprocess<K, xprec::DDouble>(kernel, safe_epsilon, n_gauss, cutoff, lmax);
    } else {
        throw std::invalid_argument("Twork must be either 'Float64' or 'Float64x2'");
    }
}


} // namespace sparseir
