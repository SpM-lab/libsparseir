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