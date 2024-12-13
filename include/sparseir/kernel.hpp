#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace sparseir {
// Forward declaration of ReducedKernel
template <typename K>
class ReducedKernel;
class AbstractSVEHints;
class SVEHintsLogistic;
class SVEHintsRegularizedBose;

/**
 * @brief Abstract base class for an integral kernel K(x, y).
 *
 * AbstractKernel represents a real binary function K(x, y) used in a Fredholm
 * integral equation of the first kind:
 *
 *     u(x) = ∫ K(x, y) v(y) dy
 *
 * where x ∈ [xmin, xmax] and y ∈ [ymin, ymax].
 *
 * In general, the kernel is applied to a scaled spectral function ρ'(y) as:
 *
 *     ∫ K(x, y) ρ'(y) dy,
 *
 * where ρ'(y) = w(y) ρ(y).
 */
class AbstractKernel {
public:
    double lambda_;
    // Constructor
    AbstractKernel(){}
    AbstractKernel(double lambda) : lambda_(lambda) {}

    /**
     * @brief Evaluate kernel at point (x, y).
     *
     * For given x, y, return the value of K(x, y). The parameters x_plus and
     * x_minus, if given, shall contain the values of x - xmin and xmax - x,
     * respectively. This is useful if either difference is to be formed and
     * cancellation is expected.
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus Optional. x - xmin.
     * @param x_minus Optional. xmax - x.
     * @return The value of K(x, y).
     */
    virtual double operator()(
        double x, double y,
        double x_plus = std::numeric_limits<double>::quiet_NaN(),
        double x_minus = std::numeric_limits<double>::quiet_NaN()) const = 0;

    /*
    sve_hints(double epsilon) const
    {
        return std::make_shared<AbstractSVEHints>(*this, epsilon);
    }
    */

    /**
     * @brief Return symmetrized kernel K(x, y) + sign * K(x, -y).
     *
     * Should be overridden by derived classes if they support symmetrization.
     *
     * @param sign The sign (+1 or -1).
     * @return A shared pointer to the symmetrized kernel.
     */
    // virtual std::shared_ptr<AbstractKernel> get_symmetrized(int sign) const
    //{
    //     throw std::runtime_error("get_symmetrized not implemented in base
    //     class");
    // }

    /**
     * @brief Return tuple (xmin, xmax) delimiting the range of allowed x
     * values.
     *
     * @return A pair containing xmin and xmax.
     */
    virtual std::pair<double, double> xrange() const
    {
        return std::make_pair(-1.0, 1.0);
    }

    /**
     * @brief Return tuple (ymin, ymax) delimiting the range of allowed y
     * values.
     *
     * @return A pair containing ymin and ymax.
     */
    virtual std::pair<double, double> yrange() const
    {
        return std::make_pair(-1.0, 1.0);
    }

    /**
     * @brief Check if the kernel is centrosymmetric.
     *
     * Returns true if and only if K(x, y) == K(-x, -y) for all values of x and
     * y. This allows the kernel to be block-diagonalized, speeding up the
     * singular value expansion by a factor of 4. Defaults to false.
     *
     * @return True if the kernel is centrosymmetric, false otherwise.
     */
    virtual bool is_centrosymmetric() const { return false; }

    /**
     * @brief Power with which the y coordinate scales.
     *
     * @return The power with which y scales.
     */
    virtual int ypower() const { return 0; }

    /**
     * @brief Convergence radius of the Matsubara basis asymptotic model.
     *
     * For improved relative numerical accuracy, the IR basis functions on the
     * Matsubara axis can be evaluated from an asymptotic expression for
     * abs(n) > conv_radius. If conv_radius is infinity, then the asymptotics
     * are unused (the default).
     *
     * @return The convergence radius.
     */
    virtual double conv_radius() const
    {
        return std::numeric_limits<double>::infinity();
    }

    /**
     * @brief Return the weight function for given statistics.
     *
     * @param statistics 'F' for fermions or 'B' for bosons.
     * @return A function representing the weight function w(y).
     */
    virtual std::function<double(double)> weight_func(char statistics) const
    {
        if (statistics != 'F' && statistics != 'B') {
            throw std::invalid_argument(
                "statistics must be 'F' for fermions or 'B' for bosons");
        }
        return [](double /*x*/) { return 1.0; };
    }

    virtual ~AbstractKernel() = default;
};

class AbstractReducedKernel : public AbstractKernel {
public:
    int sign;
    std::shared_ptr<const AbstractKernel> inner;

    // Constructor
    AbstractReducedKernel(std::shared_ptr<const AbstractKernel> inner_kernel,
                          int sign)
        : AbstractKernel(inner_kernel->lambda_),
          inner(std::move(inner_kernel)),
          sign(sign)
    {
        // Validate inputs
        if (!inner->is_centrosymmetric()) {
            throw std::invalid_argument("Inner kernel must be centrosymmetric");
        }
        if (sign != 1 && sign != -1) {
            throw std::invalid_argument("sign must be -1 or 1");
        }
    }
};

inline double callreduced(const AbstractReducedKernel &kernel, double x,
                          double y, double x_plus, double x_minus)
{
    x_plus += 1;
    auto K_plus = (*kernel.inner)(x, +y, x_plus, x_minus);
    auto K_minus = (*kernel.inner)(x, -y, x_minus, x_plus);
    return K_plus + kernel.sign * K_minus;
}

/**
 * @brief Fermionic/bosonic analytical continuation kernel.
 *
 * In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is a
 * function on [-1, 1] × [-1, 1]:
 *
 *     K(x, y) = exp(-Λ y (x + 1) / 2) / (1 + exp(-Λ y))
 *
 * LogisticKernel is a fermionic analytic continuation kernel.
 * Nevertheless, one can model the τ dependence of a bosonic correlation
 * function as follows:
 *
 *     ∫ [exp(-Λ y (x + 1) / 2) / (1 - exp(-Λ y))] ρ(y) dy = ∫ K(x, y) ρ'(y) dy,
 *
 * with ρ'(y) = w(y) ρ(y), where the weight function is given by w(y) = 1 /
 * tanh(Λ y / 2).
 */
class LogisticKernel : public AbstractKernel {
public:
    // Default constructor
    LogisticKernel() : AbstractKernel() {}

    /**
     * @brief Constructor for LogisticKernel.
     *
     * @param lambda The kernel cutoff Λ.
     */
    LogisticKernel(double lambda) : AbstractKernel(lambda)
    {
        if (lambda < 0) {
            throw std::domain_error("Kernel cutoff Λ must be non-negative");
        }
    }

    /**
     * @brief Evaluate the kernel at point (x, y).
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus Optional. x - xmin.
     * @param x_minus Optional. xmax - x.
     * @return The value of K(x, y).
     */
    double operator()(double x, double y,
                      double x_plus = std::numeric_limits<double>::quiet_NaN(),
                      double x_minus = std::numeric_limits<double>::quiet_NaN())
        const override
    {
        // Check that x and y are within the valid ranges
        std::pair<double, double> x_range = xrange();
        double xmin = x_range.first;
        double xmax = x_range.second;
        if (x < xmin || x > xmax) {
            throw std::out_of_range("x value not in range [-1, 1]");
        }
        std::pair<double, double> y_range = yrange();
        double ymin = y_range.first;
        double ymax = y_range.second;
        if (y < ymin || y > ymax) {
            throw std::out_of_range("y value not in range [-1, 1]");
        }

        std::tuple<double, double, double> uv_values =
            compute_uv(x, y, x_plus, x_minus);
        double u_plus = std::get<0>(uv_values);
        double u_minus = std::get<1>(uv_values);
        double v = std::get<2>(uv_values);

        return compute(u_plus, u_minus, v);
    }

    // Inside class LogisticKernel definition
    /*
    std::shared_ptr<SVEHintsLogistic> sve_hints(double epsilon) const
    {
        return std::make_shared<SVEHintsLogistic>(*this, epsilon);
    }
    */

    /**
     * @brief Check if the kernel is centrosymmetric.
     *
     * @return True, since LogisticKernel is centrosymmetric.
     */
    bool is_centrosymmetric() const override { return true; }

    /**
     * @brief Convergence radius of the Matsubara basis asymptotic model.
     *
     * For LogisticKernel, conv_radius = 40 * Λ.
     *
     * @return The convergence radius.
     */
    double conv_radius() const override { return 40.0 * lambda_; }

    /**
     * @brief Return the weight function for given statistics.
     *
     * @param statistics 'F' for fermions or 'B' for bosons.
     * @return A function representing the weight function w(y).
     */
    std::function<double(double)> weight_func(char statistics) const override
    {
        if (statistics == 'F') {
            // Fermionic weight function: w(y) == 1
            return [](double /*y*/) { return 1.0; };
        } else if (statistics == 'B') {
            // Bosonic weight function: w(y) == 1 / tanh(Λ*y/2)
            double lambda = lambda_;
            return [lambda](double y) {
                return 1.0 / std::tanh(0.5 * lambda * y);
            };
        } else {
            throw std::invalid_argument(
                "statistics must be 'F' for fermions or 'B' for bosons");
        }
    }

private:
    /**
     * @brief Compute the variables u_plus, u_minus, v.
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus x - xmin.
     * @param x_minus xmax - x.
     * @return A tuple containing u_plus, u_minus, and v.
     */
    std::tuple<double, double, double>
    compute_uv(double x, double y, double x_plus, double x_minus) const
    {
        // Compute u_plus, u_minus, v
        if (std::isnan(x_plus)) {
            x_plus = 1.0 + x;
        }
        if (std::isnan(x_minus)) {
            x_minus = 1.0 - x;
        }
        double u_plus = 0.5 * x_plus;
        double u_minus = 0.5 * x_minus;
        double v = lambda_ * y;
        return std::make_tuple(u_plus, u_minus, v);
    }

    /**
     * @brief Compute the kernel value using u_plus, u_minus, and v.
     *
     * @param u_plus Computed u_plus.
     * @param u_minus Computed u_minus.
     * @param v Computed v.
     * @return The value of K(x, y).
     */
    double compute(double u_plus, double u_minus, double v) const
    {
        double abs_v = std::abs(v);

        double numerator;
        double denominator;

        if (v >= 0) {
            numerator = exp_impl(-u_plus * abs_v);
        } else {
            numerator = exp_impl(-u_minus * abs_v);
        }

        denominator = 1.0 + exp_impl(-abs_v);

        return numerator / denominator;
    }
};

/**
 * @brief Regularized bosonic analytical continuation kernel.
 *
 * In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is a
 * function on [-1, 1] × [-1, 1]:
 *
 *     K(x, y) = y * exp(-Λ y (x + 1) / 2) / (exp(-Λ y) - 1)
 *
 * Care has to be taken in evaluating this expression around y = 0.
 */
class RegularizedBoseKernel : public AbstractKernel {
public:
    /**
     * @brief Constructor for RegularizedBoseKernel.
     *
     * @param lambda The kernel cutoff Λ.
     */
    explicit RegularizedBoseKernel(double lambda) : AbstractKernel(lambda)
    {
        if (lambda < 0) {
            throw std::domain_error("Kernel cutoff Λ must be non-negative");
        }
    }

    /**
     * @brief Evaluate the kernel at point (x, y).
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus Optional. x - xmin.
     * @param x_minus Optional. xmax - x.
     * @return The value of K(x, y).
     */
    double operator()(double x, double y,
                      double x_plus = std::numeric_limits<double>::quiet_NaN(),
                      double x_minus = std::numeric_limits<double>::quiet_NaN())
        const override
    {
        // Check that x and y are within the valid ranges
        std::pair<double, double> xrange_values = xrange();
        double xmin = xrange_values.first;
        double xmax = xrange_values.second;
        if (x < xmin || x > xmax) {
            throw std::out_of_range("x value not in range [-1, 1]");
        }
        std::pair<double, double> yrange_values = yrange();
        double ymin = yrange_values.first;
        double ymax = yrange_values.second;
        if (y < ymin || y > ymax) {
            throw std::out_of_range("y value not in range [-1, 1]");
        }

        std::tuple<double, double, double> uv_values =
            compute_uv(x, y, x_plus, x_minus);
        double u_plus = std::get<0>(uv_values);
        double u_minus = std::get<1>(uv_values);
        double v = std::get<2>(uv_values);

        return compute(u_plus, u_minus, v);
    }

    /**
     * @brief Check if the kernel is centrosymmetric.
     *
     * @return True, since RegularizedBoseKernel is centrosymmetric.
     */
    bool is_centrosymmetric() const override { return true; }

    /**
     * @brief Returns the power with which y scales.
     *
     * For RegularizedBoseKernel, ypower = 1.
     *
     * @return The power with which y scales.
     */
    int ypower() const override { return 1; }

    /**
     * @brief Convergence radius of the Matsubara basis asymptotic model.
     *
     * For RegularizedBoseKernel, conv_radius = 40 * Λ.
     *
     * @return The convergence radius.
     */
    double conv_radius() const override { return 40.0 * lambda_; }

    /**
     * @brief Return the weight function for given statistics.
     *
     * @param statistics 'F' for fermions or 'B' for bosons.
     * @return A function representing the weight function w(y).
     */
    std::function<double(double)> weight_func(char statistics) const override
    {
        if (statistics == 'F') {
            throw std::runtime_error(
                "Kernel is designed for bosonic functions");
        } else if (statistics == 'B') {
            // Bosonic weight function: w(y) == 1 / y
            return [](double y) { return 1.0 / y; };
        } else {
            throw std::invalid_argument(
                "statistics must be 'F' for fermions or 'B' for bosons");
        }
    }

private:
    /**
     * @brief Compute the variables u_plus, u_minus, v.
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus x - xmin.
     * @param x_minus xmax - x.
     * @return A tuple containing u_plus, u_minus, and v.
     */
    std::tuple<double, double, double>
    compute_uv(double x, double y, double x_plus, double x_minus) const
    {
        // Compute u_plus, u_minus, v
        if (std::isnan(x_plus)) {
            x_plus = 1.0 + x;
        }
        if (std::isnan(x_minus)) {
            x_minus = 1.0 - x;
        }
        double u_plus = 0.5 * x_plus;
        double u_minus = 0.5 * x_minus;
        double v = lambda_ * y;
        return std::make_tuple(u_plus, u_minus, v);
    }

    /**
     * @brief Compute the kernel value using u_plus, u_minus, and v.
     *
     * @param u_plus Computed u_plus.
     * @param u_minus Computed u_minus.
     * @param v Computed v.
     * @return The value of K(x, y).
     */
    double compute(double u_plus, double u_minus, double v) const
    {
        double absv = std::abs(v);
        double enum_val = std::exp(-absv * (v >= 0 ? u_plus : u_minus));

        // Handle the tricky expression v / (exp(v) - 1)
        double denom;
        if (absv >= 1e-200) {
            denom = absv / std::expm1(-absv);
        } else {
            denom = -1; // Assuming T is a floating-point type
        }

        return -1 / static_cast<double>(this->lambda_) * enum_val * denom;
    }
};

/**
 * @brief Restriction of centrosymmetric kernel to positive interval.
 *
 * For a kernel K on [-1, 1] × [-1, 1] that is centrosymmetric, i.e.,
 * K(x, y) = K(-x, -y), it is straightforward to show that the left/right
 * singular vectors can be chosen as either odd or even functions.
 *
 * Consequently, they are singular functions of a reduced kernel K_red on
 * [0, 1] × [0, 1] that is given as either:
 *
 *     K_red(x, y) = K(x, y) ± K(x, -y)
 *
 * This kernel is what this class represents. The full singular functions can be
 * reconstructed by (anti-)symmetrically continuing them to the negative axis.
 */
template <typename K>
class ReducedKernel : public AbstractKernel {
public:
    std::shared_ptr<const AbstractKernel>
        inner_kernel_; ///< The inner kernel K.
    int sign_;         ///< The sign (+1 or -1).

    /**
     * @brief Constructor for ReducedKernel.
     *
     * @param inner_kernel The inner kernel K.
     * @param sign The sign (+1 or -1). Must satisfy abs(sign) == 1.
     */
    ReducedKernel(std::shared_ptr<const AbstractKernel> inner_kernel, int sign)
        : AbstractKernel(inner_kernel->lambda_), // Initialize base class
          inner_kernel_(std::move(inner_kernel)),
          sign_(sign)
    {
        if (!inner_kernel_->is_centrosymmetric()) {
            throw std::invalid_argument("Inner kernel must be centrosymmetric");
        }
        if (sign != 1 && sign != -1) {
            throw std::invalid_argument("sign must be -1 or 1");
        }
    }

    /**
     * @brief Evaluate the reduced kernel at point (x, y).
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus Optional. x - xmin.
     * @param x_minus Optional. xmax - x.
     * @return The value of K_red(x, y).
     */
    double operator()(double x, double y,
                      double x_plus = std::numeric_limits<double>::quiet_NaN(),
                      double x_minus = std::numeric_limits<double>::quiet_NaN())
        const override
    {
        return call_reduced(x, y, x_plus, x_minus);
    }

    /**
     * @brief Return tuple (xmin, xmax) delimiting the range of allowed x
     * values.
     *
     * For ReducedKernel, xrange is modified to [0, xmax_inner].
     *
     * @return A pair containing xmin and xmax.
     */
    std::pair<double, double> xrange() const override
    {
        auto range = inner_kernel_->xrange();
        return std::make_pair(0.0, range.second);
    }

    /**
     * @brief Return tuple (ymin, ymax) delimiting the range of allowed y
     * values.
     *
     * For ReducedKernel, yrange is modified to [0, ymax_inner].
     *
     * @return A pair containing ymin and ymax.
     */
    std::pair<double, double> yrange() const override
    {
        auto range = inner_kernel_->yrange();
        return std::make_pair(0.0, range.second);
    }

    /**
     * @brief Check if the kernel is centrosymmetric.
     *
     * @return False, since ReducedKernel cannot be symmetrized further.
     */
    bool is_centrosymmetric() const override { return false; }

    /**
     * @brief Returns the power with which y scales.
     *
     * @return The ypower of the inner kernel.
     */
    int ypower() const override { return inner_kernel_->ypower(); }

    /**
     * @brief Convergence radius of the Matsubara basis asymptotic model.
     *
     * @return The convergence radius of the inner kernel.
     */
    double conv_radius() const override { return inner_kernel_->conv_radius(); }

private:
    /**
     * @brief Evaluate the reduced kernel.
     *
     * @param x The x value.
     * @param y The y value.
     * @param x_plus x - xmin.
     * @param x_minus xmax - x.
     * @return The value of K_red(x, y).
     */
    double call_reduced(double x, double y, double x_plus, double x_minus) const
    {
        // The reduced kernel is defined only over the interval [0, 1], which
        // means we must add one to get the x_plus for the inner kernels.
        if (std::isnan(x_plus)) {
            x_plus = 1.0 + x;
        }
        // x_minus remains the same

        // Evaluate inner kernel at (x, y) and (x, -y)
        double K_plus = inner_kernel_->operator()(x, y, x_plus, x_minus);
        double K_minus = inner_kernel_->operator()(x, -y, x_plus, x_minus);

        if (sign_ == 1) {
            return K_plus + K_minus;
        } else {
            return K_plus - K_minus;
        }
    }
};

} // namespace sparseir

namespace sparseir {
/*
    AbstractSVEHints

Discretization hints for singular value expansion of a given kernel.
*/
class AbstractSVEHints {
public:
    virtual ~AbstractSVEHints() = default;

    // Functions to compute segments for x and y
    // template <typename T = double>
    // virtual std::vector<T> segments_x() const = 0;
    // template <typename T = double>
    // virtual std::vector<T> segments_y() const = 0;

    // Additional methods if needed
    virtual int nsvals() const = 0;
    virtual int ngauss() const = 0;
};

class SVEHintsLogistic final : public AbstractSVEHints {
public:
    SVEHintsLogistic(const LogisticKernel &kernel, double epsilon)
        : kernel_(kernel), epsilon_(epsilon)
    {
    }

    template <typename T = double>
    std::vector<T> segments_x() const
    {
        int nzeros = std::max(
            static_cast<int>(std::round(15 * std::log10(kernel_.lambda_))), 1);

        // Create a range of values
        std::vector<T> temp(nzeros);
        for (int i = 0; i < nzeros; ++i) {
            temp[i] = (T)0.143 * i;
        }

        // Calculate diffs using the inverse hyperbolic cosine
        std::vector<T> diffs(nzeros);
        for (int i = 0; i < nzeros; ++i) {
            diffs[i] = 1.0 / cosh_impl(temp[i]);
        }

        // Calculate cumulative sum of diffs
        std::vector<T> zeros(nzeros);
        zeros[0] = diffs[0];
        for (int i = 1; i < nzeros; ++i) {
            zeros[i] = zeros[i - 1] + diffs[i];
        }

        // Normalize zeros
        T last_zero = zeros.back();
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] /= last_zero;
        }

        // Create the final segments vector
        std::vector<T> segments(2 * nzeros + 1, 0);
        for (int i = 0; i < nzeros; ++i) {
            segments[i] = -zeros[nzeros - i - 1];
            segments[nzeros + i + 1] = zeros[i];
        }
        return segments;
    };

    template <typename T = double>
    std::vector<T> segments_y() const
    {
        // Calculate the number of zeros
        int nzeros = std::max(
            static_cast<int>(std::round(20 * std::log10(kernel_.lambda_))), 2);

        // Initial differences
        std::vector<T> diffs = {0.01523, 0.03314, 0.04848, 0.05987, 0.06703,
                                0.07028, 0.07030, 0.06791, 0.06391, 0.05896,
                                0.05358, 0.04814, 0.04288, 0.03795, 0.03342,
                                0.02932, 0.02565, 0.02239, 0.01951, 0.01699};

        // Truncate diffs if necessary
        if (nzeros < diffs.size()) {
            diffs.resize(nzeros);
        }

        // Calculate trailing differences
        for (int i = 20; i < nzeros; ++i) {
            T x = (T)0.141 * i;
            diffs.push_back(0.25 * exp_impl(-x));
        }

        // Calculate cumulative sum of diffs
        std::vector<T> zeros(nzeros);
        zeros[0] = diffs[0];
        for (int i = 1; i < nzeros; ++i) {
            zeros[i] = zeros[i - 1] + diffs[i];
        }

        // Normalize zeros
        T last_zero = zeros.back();
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] /= last_zero;
        }
        zeros.pop_back();

        // updated nzeros
        nzeros = zeros.size();

        // Adjust zeros
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] -= 1.0;
        }

        // Create the final segments vector
        std::vector<T> segments(2 * nzeros + 3, 0);
        for (int i = 0; i < nzeros; ++i) {
            segments[1 + i] = zeros[i];
            segments[1 + nzeros + 1 + i] = -zeros[nzeros - i - 1];
        }
        segments[0] = -T(1.0);
        segments[1 + nzeros] = T(0.0);
        segments[2 * nzeros + 2] = T(1.0);
        return segments;
    }

    int nsvals() const override
    {
        double log10_Lambda = std::max(1.0, std::log10(kernel_.lambda_));
        return static_cast<int>(std::round((25 + log10_Lambda) * log10_Lambda));
    }
    int ngauss() const override
    {
        return epsilon_ >= std::sqrt(std::numeric_limits<double>::epsilon())
                   ? 10
                   : 16;
    };

private:
    const LogisticKernel &kernel_;
    double epsilon_;
};

class SVEHintsRegularizedBose : public AbstractSVEHints {
public:
    SVEHintsRegularizedBose(const RegularizedBoseKernel &kernel, double epsilon)
        : kernel_(kernel), epsilon_(epsilon)
    {
    }

    template <typename T = double>
    std::vector<T> segments_x() const
    {
        int nzeros = std::max(
            static_cast<int>(std::round(15 * std::log10(kernel_.lambda_))), 15);
        std::vector<T> temp(nzeros);
        std::vector<T> diffs(nzeros);
        std::vector<T> zeros(nzeros);

        for (int i = 0; i < nzeros; ++i) {
            temp[i] = T(0.18) * i;
            diffs[i] = 1.0 / std::cosh(temp[i]);
        }

        std::partial_sum(diffs.begin(), diffs.end(), zeros.begin());
        T last_zero = zeros.back();
        std::transform(zeros.begin(), zeros.end(), zeros.begin(),
                       [last_zero](T z) { return z / last_zero; });

        std::vector<T> result(2 * nzeros + 1, T(0));

        for (int i = 0; i < nzeros; ++i) {
            result[i] = -zeros[nzeros - i - 1];
            result[nzeros + i + 1] = zeros[i];
        }

        return result;
    }

    template <typename T = double>
    std::vector<T> segments_y() const
    {
        int nzeros = std::max(
            static_cast<int>(std::round(20 * std::log10(kernel_.lambda_))), 20);
        std::vector<T> diffs(nzeros);

        for (int j = 0; j < nzeros; ++j) {
            diffs[j] = T(0.12) / std::exp(0.0337 * j * std::log(j + 1));
        }

        // Calculate cumulative sum of diffs
        std::vector<T> zeros(nzeros);
        zeros[0] = diffs[0];
        for (int i = 1; i < nzeros; ++i) {
            zeros[i] = zeros[i - 1] + diffs[i];
        }

        // Normalize zeros
        T last_zero = zeros.back();
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] /= last_zero;
        }
        zeros.pop_back();

        // updated nzeros
        nzeros = zeros.size();

        // Adjust zeros
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] -= 1.0;
        }

        std::vector<T> result(2 * nzeros + 3, T(0));

        for (int i = 0; i < nzeros; ++i) {
            result[1 + i] = zeros[i];
            result[1 + nzeros + 1 + i] = -zeros[nzeros - i - 1];
        }
        result[0] = T(-1);
        result[1 + nzeros] = T(0);
        result[2 * nzeros + 2] = T(1);
        return result;
    }

    int nsvals() const override
    {
        double log10_Lambda = std::max(1.0, std::log10(kernel_.lambda_));
        return static_cast<int>(std::round(28 * log10_Lambda));
    }
    int ngauss() const override
    {
        return epsilon_ >= std::sqrt(std::numeric_limits<double>::epsilon())
                   ? 10
                   : 16;
    };

private:
    const RegularizedBoseKernel &kernel_;
    double epsilon_;
};

class RegularizedBoseKernelOdd : public AbstractReducedKernel {
public:
    RegularizedBoseKernelOdd(std::shared_ptr<const RegularizedBoseKernel> inner,
                             int sign)
        : AbstractReducedKernel(inner, sign)
    {
        if (!is_centrosymmetric()) {
            throw std::runtime_error("inner kernel must be centrosymmetric");
        }
        if (std::abs(sign) != 1) {
            throw std::domain_error("sign must be -1 or 1");
        }
    }

    // Implement the pure virtual function from the parent class
    double operator()(double x, double y,
                      double x_plus = std::numeric_limits<double>::quiet_NaN(),
                      double x_minus = std::numeric_limits<double>::quiet_NaN())
        const override
    {
        double v_half = inner->lambda_ * 0.5 * y;
        double xv_half = x * v_half;
        bool xy_small = xv_half < 1;
        bool sinh_range = 1e-200 < v_half && v_half < 85;
        if (xy_small && sinh_range) {
            return y * sinh_impl(xv_half) / sinh_impl(v_half);
        } else {
            return callreduced(*this, x, x, x_plus, x_minus);
        }
    }

    // You'll need to implement the isCentrosymmetric function
    // Here's a placeholder
    bool isCentrosymmetric(RegularizedBoseKernel &kernel)
    {
        // Implement this function
        return true;
    }
};

class LogisticKernelOdd : public AbstractReducedKernel {
public:
    LogisticKernelOdd(std::shared_ptr<const LogisticKernel> inner, int sign)
        : AbstractReducedKernel(inner, sign)
    {
    }
    // Implement the pure virtual function from the parent class
    double operator()(double x, double y,
                      double x_plus = std::numeric_limits<double>::quiet_NaN(),
                      double x_minus = std::numeric_limits<double>::quiet_NaN())
        const override
    {
        double v_half = inner->lambda_ * 0.5 * y;
        bool xy_small = x * v_half < 1;
        bool cosh_finite = v_half < 85;
        if (xy_small && cosh_finite) {
            return -std::sinh(v_half * x) / std::cosh(v_half);
        } else {
            return callreduced(*this, x, x, x_plus, x_minus);
        }
    }
};

inline std::shared_ptr<AbstractKernel>
get_symmetrized(std::shared_ptr<AbstractKernel> kernel, int sign)
{
    return std::make_shared<ReducedKernel<AbstractKernel>>(kernel, sign);
}

inline std::shared_ptr<AbstractKernel>
get_symmetrized(std::shared_ptr<const LogisticKernel> kernel, int sign)
{
    if (sign == -1) {
        return std::make_shared<LogisticKernelOdd>(kernel, sign);
    } else {
        return std::make_shared<ReducedKernel<LogisticKernel>>(kernel, sign);
    }
}

inline std::shared_ptr<AbstractKernel>
get_symmetrized(std::shared_ptr<const RegularizedBoseKernel> kernel, int sign)
{
    if (sign == -1) {
        return std::make_shared<RegularizedBoseKernelOdd>(kernel, sign);
    } else {
        return std::make_shared<ReducedKernel<RegularizedBoseKernel>>(kernel,
                                                                      sign);
    }
}

inline std::shared_ptr<AbstractKernel>
get_symmetrized(const LogisticKernel &kernel, int sign)
{
    auto kernel_ptr = std::make_shared<const LogisticKernel>(kernel);
    return get_symmetrized(kernel_ptr, sign);
}

inline std::shared_ptr<AbstractKernel>
get_symmetrized(const RegularizedBoseKernel &kernel, int sign)
{
    auto kernel_ptr = std::make_shared<const RegularizedBoseKernel>(kernel);
    return get_symmetrized(kernel_ptr, sign);
}

inline void get_symmetrized(AbstractReducedKernel &kernel, int sign)
{
    throw std::runtime_error("cannot symmetrize twice");
}

} // namespace sparseir

namespace sparseir {

// Function to compute matrix from Gauss rules
template <typename T>
Eigen::MatrixX<T> matrix_from_gauss(const AbstractKernel &kernel,
                                    const Rule<T> &gauss_x,
                                    const Rule<T> &gauss_y)
{

    size_t n = gauss_x.x.size();
    size_t m = gauss_y.x.size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> res(n, m);

    // Parallelize using threads
    std::vector<std::thread> threads;
    for (size_t i = 0; i < n; ++i) {
        threads.emplace_back([&, i]() {
            for (size_t j = 0; j < m; ++j) {
                // TODO: return type should be T
                double ko = kernel(static_cast<double>(gauss_x.x[i]),
                                   static_cast<double>(gauss_y.x[j]),
                                   static_cast<double>(gauss_x.x_forward[i]),
                                   static_cast<double>(gauss_x.x_backward[i]));
                res(i, j) = T(ko);
            }
        });
    }

    // Join threads
    for (auto &thread : threads) {
        thread.join();
    }

    return res;
}

class SVEHintsReduced : public AbstractSVEHints {
public:
    SVEHintsReduced(std::shared_ptr<AbstractSVEHints> inner_hints)
        : inner(inner_hints)
    {
    }

    // Implement required methods
    int nsvals() const override
    {
        // Implement this function
        // For example, you can delegate the call to the inner object
        return inner->nsvals();
    }

    int ngauss() const override
    {
        // Implement this function
        // For example, you can delegate the call to the inner object
        return inner->ngauss();
    }

private:
    std::shared_ptr<AbstractSVEHints> inner;
};

// Function to provide SVE hints
inline SVEHintsLogistic sve_hints(const LogisticKernel &kernel, double epsilon)
{
    return SVEHintsLogistic(kernel, epsilon);
}

inline SVEHintsRegularizedBose sve_hints(const RegularizedBoseKernel &kernel,
                                         double epsilon)
{
    return SVEHintsRegularizedBose(kernel, epsilon);
}

inline std::shared_ptr<AbstractSVEHints>
sve_hints(std::shared_ptr<const AbstractKernel> kernel, double epsilon)
{
    if (auto logisticKernel =
            std::dynamic_pointer_cast<const LogisticKernel>(kernel)) {
        return std::make_shared<SVEHintsLogistic>(*logisticKernel, epsilon);
    } else if (auto boseKernel =
                   std::dynamic_pointer_cast<const RegularizedBoseKernel>(
                       kernel)) {
        return std::make_shared<SVEHintsRegularizedBose>(*boseKernel, epsilon);
    } else if (auto reducedKernel =
                   std::dynamic_pointer_cast<const AbstractReducedKernel>(
                       kernel)) {
        return std::make_shared<SVEHintsReduced>(
            sve_hints(reducedKernel->inner, epsilon));
    } else {
        throw std::invalid_argument("Unsupported kernel type for SVE hints");
    }
}

inline SVEHintsReduced sve_hints(const AbstractReducedKernel &kernel,
                                 double epsilon)
{
    return SVEHintsReduced(sve_hints(kernel.inner, epsilon));
}

/*
function ngauss end
ngauss(hints::SVEHintsLogistic)        = hints.ε ≥ sqrt(eps()) ? 10 : 16
ngauss(hints::SVEHintsRegularizedBose) = hints.ε ≥ sqrt(eps()) ? 10 : 16
ngauss(hints::SVEHintsReduced)         = ngauss(hints.inner_hints)
*/

/*
std::shared_ptr<AbstractSVEHints> sve_hints(const AbstractReducedKernel& kernel,
double epsilon) {
    // Assume kernel.inner is a method that returns the inner kernel
    auto innerHints = sve_hints(kernel.inner(), epsilon);
    return std::make_shared<SVEHintsReduced>(innerHints);
}
*/
} // namespace sparseir