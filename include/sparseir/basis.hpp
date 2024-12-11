// Template class for FiniteTempBasis
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sparseir {

template <typename T>
class AbstractBasis {
public:
    virtual ~AbstractBasis() { }

    /**
     * @brief Basis functions on the imaginary time axis.
     *
     * Set of IR basis functions on the imaginary time (tau) axis, where tau
     * is a real number between zero and beta. To get the l-th basis function
     * at imaginary time tau, use:
     *
     *     T ul_tau = u(l, tau);
     *
     * @param l Index of the basis function.
     * @param tau Imaginary time variable.
     * @return Value of the l-th basis function at time tau.
     */
    virtual T u(int l, T tau) const = 0;

    /**
     * @brief Basis functions on the reduced Matsubara frequency axis.
     *
     * Set of IR basis functions on the reduced Matsubara frequency (wn) axis,
     * where wn is an integer. These are related to u by the Fourier transform:
     *
     *     uhat(n) = ∫₀^β dτ exp(iπnτ/β) * u(τ)
     *
     * To get the l-th basis function at some reduced frequency wn, use:
     *
     *     T uhat_l_n = uhat(l, wn);
     *
     * @param l Index of the basis function.
     * @param wn Reduced Matsubara frequency (integer multiplier).
     * @return Value of the l-th basis function at frequency wn.
     */
    virtual T uhat(int l, int wn) const = 0;

    /**
     * @brief Quantum statistic ("F" for fermionic, "B" for bosonic).
     *
     * @return Character representing the quantum statistics.
     */
    virtual char statistics() const = 0;

    /**
     * @brief Access basis functions/singular values for given index/indices.
     *
     * This can be used to truncate the basis to the n most significant
     * singular values: `basis[0, n]`.
     *
     * @param index Index or range of indices.
     * @return Pointer to the truncated basis (implementation-defined).
     */
    virtual AbstractBasis<T> *operator[](int index) const = 0;

    /**
     * @brief Shape of the basis function set.
     *
     * @return Pair representing the shape (rows, columns).
     */
    virtual std::pair<int, int> shape() const = 0;

    /**
     * @brief Number of basis functions / singular values.
     *
     * @return Size of the basis function set.
     */
    virtual int size() const = 0;

    /**
     * @brief Significances of the basis functions.
     *
     * Vector of significance values, one for each basis function. Each value
     * is a number between 0 and 1 which provides an a-priori bound on the
     * relative error made by discarding the associated coefficient.
     *
     * @return Vector of significance values.
     */
    virtual const std::vector<T> &significance() const = 0;

    /**
     * @brief Accuracy of the basis.
     *
     * Upper bound to the relative error of representing a propagator with
     * the given number of basis functions (number between 0 and 1).
     *
     * @return Accuracy value.
     */
    virtual T accuracy() const
    {
        const auto &sig = significance();
        return !sig.empty() ? sig.back() : static_cast<T>(0);
    }

    /**
     * @brief Basis cutoff parameter, Λ == β * wmax, or NaN if not present.
     *
     * @return Cutoff parameter Λ.
     */
    virtual T lambda() const = 0;

    /**
     * @brief Inverse temperature.
     *
     * @return Value of β.
     */
    virtual T beta() const = 0;

    /**
     * @brief Real frequency cutoff or NaN if not present.
     *
     * @return Maximum real frequency wmax.
     */
    virtual T wmax() const = 0;

    /**
     * @brief Default sampling points on the imaginary time axis.
     *
     * @param npoints Minimum number of sampling points to return.
     * @return Vector of sampling points on the τ-axis.
     */
    virtual std::vector<T>
    default_tau_sampling_points(int npoints = 0) const = 0;

    /**
     * @brief Default sampling points on the imaginary frequency axis.
     *
     * @param npoints Minimum number of sampling points to return.
     * @param positive_only If true, only return non-negative frequencies.
     * @return Vector of sampling points on the Matsubara axis.
     */
    virtual std::vector<int>
    default_matsubara_sampling_points(int npoints = 0,
                                      bool positive_only = false) const = 0;

    /**
     * @brief Returns true if the sampling is expected to be well-conditioned.
     *
     * @return True if well-conditioned.
     */
    virtual bool is_well_conditioned() const { return true; }
};

} // namespace sparseir

namespace sparseir {

template <typename S, typename K>
class FiniteTempBasis : public AbstractBasis<S> {
public:
    K kernel;
    SVEResult<K> sve_result;
    double accuracy;
    double beta; // β
    PiecewiseLegendrePolyVector u;
    PiecewiseLegendrePolyVector v;
    std::vector<double> s;
    PiecewiseLegendreFTVector<S> uhat;
    PiecewiseLegendreFTVector<S> uhat_full;

    // Constructor
    FiniteTempBasis(S statistics, double beta, double omega_max,
                    double epsilon = 0.0, int max_size = -1, K kernel = K(),
                    SVEResult<K> sve_result = SVEResult<K>())
        : kernel(kernel), sve_result(sve_result), beta(beta)
    {
        if (beta <= 0.0)
            throw std::domain_error("Inverse temperature β must be positive");
        if (omega_max < 0.0)
            throw std::domain_error(
                "Frequency cutoff ωmax must be non-negative");

        // Partition the SVE result
        auto part_result = sve_result.part(epsilon, max_size);
        auto u_ = std::get<0>(part_result);
        auto s_ = std::get<1>(part_result);
        auto v_ = std::get<2>(part_result);

        int L = static_cast<int>(s_.size());

        // Calculate accuracy
        if (sve_result.s.size() > s_.size()) {
            accuracy = sve_result.s[s_.size()] / sve_result.s[0];
        } else {
            accuracy = sve_result.s.back() / sve_result.s[0];
        }

        // Scaling variables
        omega_max = kernel.Lambda() / beta;
        Eigen::VectorXd u_knots = (beta / 2.0) * (u_.knots.array() + 1.0);
        Eigen::VectorXd v_knots = omega_max * v_.knots;

        u = PiecewiseLegendrePolyVector(u_, u_knots, (beta / 2.0) * u_.delta_x,
                                        u_.symmetry);
        v = PiecewiseLegendrePolyVector(v_, v_knots, omega_max * v_.delta_x,
                                        v_.symmetry);

        // Scale singular values
        double scale_factor = std::sqrt(beta / 2.0 * omega_max) *
                              std::pow(omega_max, -kernel.getYPower());
        s.resize(L);
        for (int i = 0; i < L; ++i)
            s[i] = scale_factor * s_[i];

        // Fourier transforms scaling
        PiecewiseLegendrePolyVector u_base_full(
            std::sqrt(beta) * sve_result.u.data, sve_result.u);
        uhat_full = PiecewiseLegendreFTVector<S>(u_base_full, statistics,
                                                 kernel.convRadius());
        uhat = uhat_full.slice(0, L);
    }

    // Show function (for printing)
    void show() const
    {
        std::cout << s.size() << "-element FiniteTempBasis<" << typeid(S).name()
                  << "> with "
                  << "β = " << beta << ", ωmax = " << getOmegaMax()
                  << " and singular values:\n";
        for (size_t i = 0; i < s.size() - 1; ++i)
            std::cout << " " << s[i] << "\n";
        std::cout << " " << s.back() << "\n";
    }

    // Overload operator[] for indexing (get a subset of the basis)
    FiniteTempBasis<S, K> operator[](const std::pair<int, int> &range) const
    {
        int new_size = range.second - range.first + 1;
        return FiniteTempBasis<S, K>(statistics(), beta, getOmegaMax(), 0.0,
                                     new_size, kernel, sve_result);
    }

    // Calculate significance
    Eigen::VectorXd significance() const
    {
        Eigen::VectorXd s_vec =
            Eigen::Map<const Eigen::VectorXd>(s.data(), s.size());
        return s_vec / s_vec[0];
    }

    // Getter for accuracy
    double getAccuracy() const { return accuracy; }

    // Getter for ωmax
    double getOmegaMax() const { return kernel.Lambda() / beta; }

    // Getter for SVEResult
    const SVEResult<K> &getSVEResult() const { return sve_result; }

    // Getter for kernel
    const K &getKernel() const { return kernel; }

    // Getter for Λ
    double Lambda() const { return kernel.Lambda(); }

    // Default τ sampling points
    Eigen::VectorXd defaultTauSamplingPoints() const
    {
        Eigen::VectorXd x =
            default_sampling_points(sve_result.u, static_cast<int>(s.size()));
        return (beta / 2.0) * (x.array() + 1.0);
    }

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
            default_sampling_points(sve_result.v, static_cast<int>(s.size()));
        return getOmegaMax() * y.array();
    }

    // Rescale function
    FiniteTempBasis<S, K> rescale(double new_beta) const
    {
        double new_omega_max = kernel.Lambda() / new_beta;
        return FiniteTempBasis<S, K>(statistics(), new_beta, new_omega_max, 0.0,
                                     static_cast<int>(s.size()), kernel,
                                     sve_result);
    }

private:
    // Placeholder statistics function
    S statistics() const { return S(); }

    // Default sampling points function
    Eigen::VectorXd
    default_sampling_points(const PiecewiseLegendrePolyVector &u, int L) const
    {
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
            x0.tail(maxima.size()) = maxima;
            x0[x0.size() - 1] = right;

            if (x0.size() != L) {
                std::cerr << "Warning: Expected " << L
                          << " sampling points, got " << x0.size() << ".\n";
            }

            return x0;
        }
    }

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

} // namespace sparseir
