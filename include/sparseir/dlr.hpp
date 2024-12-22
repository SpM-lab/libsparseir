#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace sparseir {

// Forward declaration of AbstractBasis
// class AbstractBasis;

template <typename Stats>
struct MatsubaraPoles
{
    double beta;
    std::vector<double> poles;

    MatsubaraPoles(double beta, const std::vector<double> &poles)
        : beta(beta), poles(poles)
    {
    }

    // Compute the imaginary part of the frequency
    double valueim(int n) const
    {
        return M_PI * (2 * n + Stats().zeta()) / beta;
    }

    std::vector<std::complex<double>> operator()(int n) const
    {
        return evaluate_frequency(n);
    }

    // TODO: Implement this
    // Eigen::MatrixXd operator()(const std::vector<int> &n) const

    // Generic methods to be specialized
    std::vector<std::complex<double>> evaluate_frequency(int n) const;
    // std::vector<double> evaluate_time(double tau) const;
};

// Specialization for Fermionic evaluate_frequency
template <>
inline std::vector<std::complex<double>>
MatsubaraPoles<Fermionic>::evaluate_frequency(int n) const
{
    std::vector<std::complex<double>> result(poles.size());
    double freq = valueim(n);
    for (size_t i = 0; i < poles.size(); ++i) {
        result[i] = 1.0 / (std::complex<double>(0, freq) - poles[i]);
    }
    return result;
}

// Specialization for Bosonic evaluate_frequency
template <>
inline std::vector<std::complex<double>>
MatsubaraPoles<Bosonic>::evaluate_frequency(int n) const
{
    std::vector<std::complex<double>> result(poles.size());
    double freq = valueim(n);
    for (size_t i = 0; i < poles.size(); ++i) {
        result[i] = std::tanh(beta / 2 * poles[i]) /
                    (std::complex<double>(0, freq) - poles[i]);
    }
    return result;
}

// Template class for TauPoles
template <typename S>
class TauPoles {
public:
    double beta;               // Corresponds to β in Julia
    std::vector<double> poles; // Poles
    double omega_max;          // Maximum absolute pole value

    // Constructor
    TauPoles(double beta, const std::vector<double> &poles)
        : beta(beta), poles(poles), omega_max(calculate_omega_max(poles))
    {
    }

    // Utility function to compute maximum absolute value
    static double calculate_omega_max(const std::vector<double> &poles)
    {
        return *std::max_element(
            poles.begin(), poles.end(),
            [](double a, double b) { return std::abs(a) < std::abs(b); });
    }

    // operator() to evaluate for a vector of τ
    Eigen::MatrixXd operator()(const std::vector<double> &tau) const
    {
        // Validate that τ is in the range [0, β]
        for (double t : tau) {
            if (t < 0 || t > beta) {
                throw std::domain_error("τ must be in [0, β], found " +
                                        std::to_string(t) + " outside of [0, " +
                                        std::to_string(beta) + "]");
            }
        }

        // Compute x and y
        std::vector<double> x(tau.size());
        for (size_t i = 0; i < tau.size(); ++i) {
            x[i] = 2.0 * tau[i] / beta - 1.0;
        }

        std::vector<double> y(poles.size());
        for (size_t i = 0; i < poles.size(); ++i) {
            y[i] = poles[i] / omega_max;
        }

        double lambda = beta * omega_max;
        LogisticKernel logistic_kernel(lambda);

        // Compute result
        Eigen::MatrixXd result(poles.size(), tau.size());
        for (size_t i = 0; i < tau.size(); ++i) {
            for (size_t j = 0; j < poles.size(); ++j) {
                result(j, i) = -logistic_kernel(x[i], y[j]);
            }
        }

        return result;
    }
};

/*
inline std::vector<double> Poles<Fermionic>::evaluate_time(double tau) const
{
    std::vector<double> result(poles.size());
    for (size_t i = 0; i < poles.size(); ++i) {
        result[i] =
            -std::exp(-poles[i] * tau) / (1 + std::exp(-beta * poles[i]));
    }
    return result;
}

// Specialization for Bosonic evaluate_time
template <>
inline std::vector<double> Poles<Bosonic>::evaluate_time(double tau) const
{
    std::vector<double> result(poles.size());
    for (size_t i = 0; i < poles.size(); ++i) {
        result[i] =
            std::exp(-poles[i] * tau) / (1 - std::exp(-beta * poles[i]));
    }
    return result;
}
*/

} // namespace sparseir