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

// Abstract class with S = Fermionic or Bosonic
template <typename S>
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
    // virtual S u(int l, double tau) const = 0;

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
    // virtual S uhat(int l, int wn) const = 0;

    /**
     * @brief Quantum statistic ("F" for fermionic, "B" for bosonic).
     *
     * @return Character representing the quantum statistics.
     */
    // virtual S statistics() const = 0;

    /**
     * @brief Access basis functions/singular values for given index/indices.
     *
     * This can be used to truncate the basis to the n most significant
     * singular values: `basis[0, n]`.
     *
     * @param index Index or range of indices.
     * @return Pointer to the truncated basis (implementation-defined).
     */
    // virtual AbstractBasis<S> *operator[](int index) const = 0;

    /**
     * @brief Shape of the basis function set.
     *
     * @return Pair representing the shape (rows, columns).
     */
    // virtual std::pair<int, int> shape() const = 0;

    /**
     * @brief Number of basis functions / singular values.
     *
     * @return Size of the basis function set.
     */
    // virtual int size() const = 0;

    /**
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
    virtual double accuracy() const
    {
        const Eigen::VectorXd sig = significance();
        return sig.size() > 0 ? sig(sig.size() - 1) : static_cast<double>(0);
    }

    /**
     * @brief Basis cutoff parameter, Λ == β * wmax, or NaN if not present.
     *
     * @return Cutoff parameter Λ.
     */
    // virtual double lambda() const = 0;

    /**
     * @brief Inverse temperature.
     *
     * @return Value of β.
     */
    // virtual double beta() const = 0;

    /**
     * @brief Real frequency cutoff or NaN if not present.
     *
     * @return Maximum real frequency wmax.
     */
    // virtual double wmax() const = 0;

    /**
     * @brief Default sampling points on the imaginary time axis.
     *
     * @param npoints Minimum number of sampling points to return.
     * @return Vector of sampling points on the τ-axis.
     */
    // virtual Eigen::VectorXd
    // default_tau_sampling_points(int npoints = 0) const = 0;

    /**
     * @brief Default sampling points on the imaginary frequency axis.
     *
     * @param npoints Minimum number of sampling points to return.
     * @param positive_only If true, only return non-negative frequencies.
     * @return Vector of sampling points on the Matsubara axis.
     */
    // virtual Eigen::VectorXd
    // default_matsubara_sampling_points(int npoints = 0,
    //                                   bool positive_only = false) const = 0;

    /**
     * @brief Returns true if the sampling is expected to be well-conditioned.
     *
     * @return True if well-conditioned.
     */
    // virtual bool is_well_conditioned() const { return true; }
};

} // namespace sparseir

namespace sparseir {

template <typename S, typename K=LogisticKernel<DDouble>>
class FiniteTempBasis : public AbstractBasis<S> {
public:
    std::shared_ptr<K> kernel;
    std::shared_ptr<SVEResult> sve_result;
    double accuracy;
    double beta;
    PiecewiseLegendrePolyVector u;
    PiecewiseLegendrePolyVector v;
    Eigen::VectorXd s;
    PiecewiseLegendreFTVector<S> uhat;
    PiecewiseLegendreFTVector<S> uhat_full;

    FiniteTempBasis(double beta, double omega_max, double epsilon,
               std::shared_ptr<K> kernel, SVEResult sve_result, int max_size = -1)
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
        this->kernel = kernel;
        this->sve_result = std::make_shared<SVEResult>(sve_result);

        double wmax = this->kernel->lambda_ / beta;

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

        this->s = (std::sqrt(beta / 2 * wmax) * std::pow(wmax, -(this->kernel->ypower()))) * s_;

        Eigen::Tensor<double, 3> udata3d = sve_result.u.get_data();
        PiecewiseLegendrePolyVector uhat_base_full =
            PiecewiseLegendrePolyVector(sqrt(beta) * udata3d, sve_result.u);
        S statistics = S();

        this->uhat_full = PiecewiseLegendreFTVector<S>(
            uhat_base_full, statistics, kernel->conv_radius());

        std::vector<PiecewiseLegendreFT<S>> uhat_polyvec;
        for (int i = 0; i < this->s.size(); ++i) {
            uhat_polyvec.push_back(this->uhat_full[i]);
            }
        this->uhat = PiecewiseLegendreFTVector<S>(uhat_polyvec);
    }

    // Delegating constructor 1
    FiniteTempBasis(double beta, double omega_max, double epsilon, int max_size=-1)
        : FiniteTempBasis(beta, omega_max, epsilon, std::make_shared<K>(beta * omega_max),
                        compute_sve<typename K::ScalarT>(std::make_shared<K>(beta * omega_max), epsilon),
                        max_size){}

    // Delegating constructor 2
    FiniteTempBasis(double beta, double omega_max,
                    double epsilon, std::shared_ptr<K> kernel)
        : FiniteTempBasis(beta, omega_max, epsilon,
                      kernel,
                      compute_sve<typename K::ScalarT>(kernel, epsilon),
                      -1){}

    // Overload operator[] for indexing (get a subset of the basis)
    FiniteTempBasis<S, K> operator[](const std::pair<int, int> &range) const
    {
        int new_size = range.second - range.first + 1;
        return FiniteTempBasis<S, K>(statistics(), beta, get_wmax(), 0.0,
                                     new_size, kernel, sve_result);
    }

    // Calculate significance
    const Eigen::VectorXd significance() const override { return s / s[0]; }

    // Getter for accuracy
    double get_accuracy() const { return accuracy; }

    // Getter for ωmax
    double get_wmax() const { return kernel->lambda_ / beta; }

    // Getter for SVEResult
    std::shared_ptr<SVEResult> &getSVEResult() const { return sve_result; }

    // Getter for kernel
    std::shared_ptr<K> &getKernel() const { return kernel; }

    // Getter for Λ
    double Lambda() const { return kernel->lambda_; }

    // Default τ sampling points
    Eigen::VectorXd defaultTauSamplingPoints() const
    {
        Eigen::VectorXd x =
            default_sampling_points(sve_result->u, static_cast<int>(s.size()));
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
            default_sampling_points(sve_result->v, static_cast<int>(s.size()));
        return get_wmax() * y.array();
    }

    // Rescale function
    FiniteTempBasis<S, K> rescale(double new_beta) const
    {
        double new_omega_max = kernel->lambda_ / new_beta;
        return FiniteTempBasis<S, K>(
            new_beta,
            new_omega_max,
            std::numeric_limits<double>::quiet_NaN(),
            kernel,
            *sve_result,
            static_cast<int>(s.size())
        );
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
            x0.tail(maxima.size()) = maxima;
            x0[x0.size() - 1] = right;

            if (x0.size() != L) {
                std::cerr << "Warning: Expected " << L
                          << " sampling points, got " << x0.size() << ".\n";
            }

            return x0;
        }
    }

template <typename S>
inline Eigen::VectorXd default_tau_sampling_points(std::shared_ptr<FiniteTempBasis<S>> basis){
    int sz = basis.sve_result.s.size();
    auto x = default_samplint_points(basis.sve_result.u, sz);
    return (basis.beta / 2.0) * (x.array() + 1.0);
}

template<typename T>
std::pair<FiniteTempBasis<Fermionic, LogisticKernel<T>>,
                 FiniteTempBasis<Bosonic, LogisticKernel<T>>>
    finite_temp_bases(
        double beta, double omega_max,
        double epsilon,
        SVEResult sve_result
    )
{
    auto kernel = std::make_shared<LogisticKernel<T>>(beta * omega_max);
    auto basis_f = FiniteTempBasis<Fermionic, LogisticKernel<T>>(
        beta, omega_max, epsilon, kernel, sve_result);
    auto basis_b = FiniteTempBasis<Bosonic, LogisticKernel<T>>(
        beta, omega_max, epsilon, kernel, sve_result);
    return std::make_pair(basis_f, basis_b);
}

template<typename T>
std::pair<FiniteTempBasis<Fermionic, LogisticKernel<T>>,
                 FiniteTempBasis<Bosonic, LogisticKernel<T>>>
    finite_temp_bases(
        double beta, double omega_max,
        double epsilon = std::numeric_limits<double>::quiet_NaN()
    )
{
    auto kernel = std::make_shared<LogisticKernel<T>>(beta * omega_max);
    SVEResult sve_result = compute_sve<T>(kernel, epsilon);
    auto basis_f = FiniteTempBasis<Fermionic, LogisticKernel<T>>(
        beta, omega_max, epsilon, kernel, sve_result);
    auto basis_b = FiniteTempBasis<Bosonic, LogisticKernel<T>>(
        beta, omega_max, epsilon, kernel, sve_result);
    return std::make_pair(basis_f, basis_b);
}
} // namespace sparseir
