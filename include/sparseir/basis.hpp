// Template class for FiniteTempBasis
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <memory> // for std::shared_ptr
#include <vector>

namespace sparseir {

Eigen::VectorXd default_sampling_points(const PiecewiseLegendrePolyVector &u, int L);

// Abstract class with S = Fermionic or Bosonic
template <typename S>
class AbstractBasis {
public:
    virtual ~AbstractBasis() { }
    double beta;
    double get_beta() const { return beta; }
    /*
     * @brief Significances of the basis functions.
     *
     * Vector of significance values, one for each basis function. Each value
     * is a number between 0 and 1 which provides an a-priori bound on the
     * relative error made by discarding the associated coefficient.
     *
     * @return Vector of significance values.
     */
    virtual const Eigen::VectorXd significance() const = 0;

    /**
     * @brief Accuracy of the basis.
     *
     * Upper bound to the relative error of representing a propagator with
     * the given number of basis functions (number between 0 and 1).
     *
     * @return Accuracy value.
     */
    virtual double get_accuracy() const = 0;
    virtual double get_wmax() const = 0;
    virtual size_t size() const = 0;
    virtual const Eigen::VectorXd default_tau_sampling_points() const = 0;

};

} // namespace sparseir

namespace sparseir {

template <typename S>
class FiniteTempBasis : public AbstractBasis<S> {
public:
    double lambda;
    std::shared_ptr<SVEResult> sve_result;
    double accuracy;
    //double beta;
    PiecewiseLegendrePolyVector u;
    PiecewiseLegendrePolyVector v;
    Eigen::VectorXd s;
    PiecewiseLegendreFTVector<S> uhat;
    PiecewiseLegendreFTVector<S> uhat_full;

    FiniteTempBasis(double beta, double omega_max, double epsilon,
               const LogisticKernel &kernel, SVEResult sve_result, int max_size = -1)
    {
        if (sve_result.s.size() == 0) {
            throw std::runtime_error("SVE result sve_result.s is empty");
        }
        if (beta <= 0.0) {
            throw std::domain_error(
                "Inverse temperature beta must be positive");
        }
        if (omega_max < 0.0) {
            throw std::domain_error(
                "Frequency cutoff omega_max must be non-negative");
        }
        this->beta = beta;
        this->lambda = kernel.lambda_;
        this->sve_result = std::make_shared<SVEResult>(sve_result);

        double wmax = this->lambda / beta;

        auto part_result = sve_result.part(epsilon, max_size);
        PiecewiseLegendrePolyVector u_ = std::get<0>(part_result);
        Eigen::VectorXd s_ = std::get<1>(part_result);
        PiecewiseLegendrePolyVector v_ = std::get<2>(part_result);
        double sve_result_s0 = sve_result.s(0);

        if (sve_result.s.size() > s_.size()) {
            this->accuracy = sve_result.s(s_.size()) / sve_result_s0;
        } else {
            this->accuracy = sve_result.s(s_.size() - 1) / sve_result_s0;
        }

        /*
        Port the following Julia code to C++:
        # The polynomials are scaled to the new variables by transforming
        the # knots according to: tau = β/2 * (x + 1), w = ωmax * y. Scaling # the
        data is not necessary as the normalization is inferred. ωmax = Λ(kernel) / β
        u_knots = (β / 2) .* (knots(u_) .+ 1)
        v_knots = ωmax .* knots(v_)
        u = PiecewiseLegendrePolyVector(u_, u_knots; Δx=(β / 2) .* Δx(u_),
        symm=symm(u_)) v = PiecewiseLegendrePolyVector(v_, v_knots; Δx=ωmax .*
        Δx(v_), symm=symm(v_))
        */

        auto u_knots_ = u_.polyvec[0].knots;
        auto v_knots_ = v_.polyvec[0].knots;
        std::cout << "beta: " << beta << std::endl;
        Eigen::VectorXd u_knots = (beta / 2) * (u_knots_.array() + 1);
        Eigen::VectorXd v_knots = wmax * v_knots_;

        Eigen::VectorXd deltax4u = (beta / 2) * diff(u_knots);
        Eigen::VectorXd deltax4v = wmax * diff(v_knots);
        std::vector<int> u_symm_vec;
        for (int i = 0; i < u_.size(); ++i) {
            u_symm_vec.push_back(u_.polyvec[i].get_symm());
        }
        std::vector<int> v_symm_vec;
        for (int i = 0; i < v_.size(); ++i) {
            v_symm_vec.push_back(v_.polyvec[i].get_symm());
        }

        Eigen::VectorXi u_symm = Eigen::Map<Eigen::VectorXi>(u_symm_vec.data(), u_symm_vec.size());
        Eigen::VectorXi v_symm = Eigen::Map<Eigen::VectorXi>(v_symm_vec.data(), v_symm_vec.size());

        PiecewiseLegendrePolyVector u_new = PiecewiseLegendrePolyVector(u_, u_knots, deltax4u, u_symm);
        PiecewiseLegendrePolyVector v_new = PiecewiseLegendrePolyVector(v_, v_knots, deltax4v, v_symm);
        this->u = u_new;
        this->v = v_new;
        this->s = (std::sqrt(beta / 2 * wmax) * std::pow(wmax, -(kernel.ypower()))) * s_;

        Eigen::Tensor<double, 3> udata3d = sve_result.u.get_data();
        PiecewiseLegendrePolyVector uhat_base_full =
            PiecewiseLegendrePolyVector(sqrt(beta) * udata3d, sve_result.u);
        S statistics = S();

        this->uhat_full = PiecewiseLegendreFTVector<S>(
            uhat_base_full, statistics, kernel.conv_radius());

        std::vector<PiecewiseLegendreFT<S>> uhat_polyvec;
        for (int i = 0; i < this->s.size(); ++i) {
            uhat_polyvec.push_back(this->uhat_full[i]);
            }
        this->uhat = PiecewiseLegendreFTVector<S>(uhat_polyvec);
    }

    // Delegating constructor 1
    FiniteTempBasis(double beta, double omega_max, double epsilon,
                    int max_size = -1)
        : FiniteTempBasis(
              beta, omega_max, epsilon, LogisticKernel(beta * omega_max),
              compute_sve(LogisticKernel(beta * omega_max), epsilon), max_size)
    {
    }

    // Delegating constructor 2
    FiniteTempBasis(double beta, double omega_max,
                    double epsilon, const LogisticKernel& kernel)
        : FiniteTempBasis(beta, omega_max, epsilon,
                      kernel,
                      compute_sve(kernel, epsilon),
                      -1){}

    // Overload operator[] for indexing (get a subset of the basis)
    FiniteTempBasis<S> operator[](const std::pair<int, int> &range) const
    {
        int new_size = range.second - range.first + 1;
        return FiniteTempBasis<S>(statistics(), this->get_beta(), get_wmax(), 0.0,
                                     new_size, lambda, sve_result);
    }

    // Calculate significance
    const Eigen::VectorXd significance() const override { return s / s[0]; }

    // Getter for accuracy
    double get_accuracy() const override { return accuracy; }

    // Getter for ωmax
    double get_wmax() const override { return lambda / this->get_beta(); }
    size_t size() const override { return s.size(); }
    // Getter for SVEResult
    std::shared_ptr<const SVEResult> getSVEResult() const { return sve_result; }

    // Getter for kernel
    //std::shared_ptr<const K> getKernel() const { return kernel; }

    // Getter for Λ
    //double Lambda() const { return lambda; }

    // Default Matsubara sampling points
    Eigen::VectorXd
    defaultMatsubaraSamplingPoints(bool positive_only = false) const
    {
        return defaultMatsubaraSamplingPoints(
            uhat_full, static_cast<int>(s.size()), false, positive_only);
    }

    // Default ω sampling points
    Eigen::VectorXd defaultOmegaSamplingPoints() const
    {
        Eigen::VectorXd y =
            default_sampling_points(sve_result->v, static_cast<int>(s.size()));
        return this->get_wmax() * y.array();
    }

    // Rescale function
    FiniteTempBasis<S> rescale(double new_beta) const
    {
        double new_omega_max = lambda / new_beta;
        return FiniteTempBasis<S>(
            new_beta, new_omega_max, std::numeric_limits<double>::quiet_NaN(),
            lambda, *sve_result, static_cast<int>(s.size()));
    }

    const Eigen::VectorXd default_tau_sampling_points() const override {
        int sz = this->sve_result->s.size();
        auto x = default_sampling_points(this->sve_result->u, sz);
        return (this->beta / 2.0) * (x.array() + 1.0);
    }

private:
    // Placeholder statistics function
    std::shared_ptr<S> statistics() const { return std::make_shared<S>(); }

    // Default Matsubara sampling points function
    Eigen::VectorXd defaultMatsubaraSamplingPoints(
        const PiecewiseLegendreFTVector<S> &u_hat_full, int L,
        bool fence = false, bool positive_only = false) const
    {
        int l_requested = L;

        // Adjust l_requested based on statistics
        if (std::is_same<S, Fermionic>::value && l_requested % 2 != 0)
            l_requested += 1;
        else if (std::is_same<S, Bosonic>::value && l_requested % 2 == 0)
            l_requested += 1;

        Eigen::VectorXd omega_n;

        if (l_requested < u_hat_full.size()) {
            omega_n = u_hat_full[l_requested + 1].signChanges(positive_only);
        } else {
            // Use extrema as a fallback
            omega_n = u_hat_full.back().findExtrema(positive_only);
            if (std::is_same<S, Bosonic>::value) {
                omega_n.conservativeResize(omega_n.size() + 1);
                omega_n[omega_n.size() - 1] = 0.0;
                std::sort(omega_n.data(), omega_n.data() + omega_n.size());
                omega_n = omega_n.unaryExpr(
                    [](double x) { return std::unique(&x, &x + 1); });
            }
        }

        int expected_size = l_requested;
        if (positive_only)
            expected_size = (expected_size + 1) / 2;

        if (omega_n.size() != expected_size) {
            std::cerr << "Warning: Requested " << expected_size
                      << " sampling frequencies for basis size L = " << L
                      << ", but got " << omega_n.size() << ".\n";
        }

        if (fence)
            fenceMatsubaraSamplingPoints(omega_n, positive_only);

        return omega_n;
    }

    // Fence Matsubara sampling points
    void fenceMatsubaraSamplingPoints(Eigen::VectorXd &omega_n,
                                      bool positive_only) const
    {
        // Implement fencing logic here...
    }
};

    // Default sampling points function
inline Eigen::VectorXd default_sampling_points(const PiecewiseLegendrePolyVector &u, int L) {
    if (u.xmin() != -1.0 || u.xmax() != 1.0)
        throw std::runtime_error("Expecting unscaled functions here.");

    if (L < u.size()) {
        // TODO: Resolve this errors.
        return u.polyvec[L].roots();
    } else {
        // Approximate roots by extrema
        // TODO: resolve this error
        PiecewiseLegendrePoly poly = u.polyvec.back();
        Eigen::VectorXd maxima = poly.deriv().roots();

        double left = (maxima[0] + poly.xmin) / 2.0;
        double right = (maxima[maxima.size() - 1] + poly.xmax) / 2.0;

        Eigen::VectorXd x0(maxima.size() + 2);
        x0[0] = left;
        x0.segment(1, maxima.size()) = maxima;
        x0[x0.size() - 1] = right;

        if (x0.size() != L) {
            std::cerr << "Warning: Expected " << L
                        << " sampling points, got " << x0.size() << ".\n";
        }

        return x0;
    }
}

inline std::pair<FiniteTempBasis<Fermionic>,
                 FiniteTempBasis<Bosonic>>
    finite_temp_bases(
        double beta, double omega_max,
        double epsilon,
        SVEResult sve_result
    )
{
    auto kernel = std::make_shared<LogisticKernel>(beta * omega_max);
    auto basis_f = FiniteTempBasis<Fermionic>(
        beta, omega_max, epsilon, *kernel, sve_result);
    auto basis_b = FiniteTempBasis<Bosonic>(
        beta, omega_max, epsilon, *kernel, sve_result);
    return std::make_pair(basis_f, basis_b);
}

inline std::pair<FiniteTempBasis<Fermionic>,
                 FiniteTempBasis<Bosonic>>
    finite_temp_bases(
        double beta, double omega_max,
        double epsilon = std::numeric_limits<double>::quiet_NaN()
    )
{
    auto kernel = LogisticKernel(beta * omega_max);
    SVEResult sve_result = compute_sve(kernel, epsilon);
    auto basis_f = FiniteTempBasis<Fermionic>(
        beta, omega_max, epsilon, kernel, sve_result);
    auto basis_b = FiniteTempBasis<Bosonic>(
        beta, omega_max, epsilon, kernel, sve_result);
    return std::make_pair(basis_f, basis_b);
}

} // namespace sparseir
