#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

namespace sparseir {

using Eigen::Dynamic;
using Eigen::Matrix;

template <typename T>
T legval(T x, const std::vector<T> &c)
{
    int nd = c.size();
    if (nd < 2) {
        return c.back();
    }

    T c0 = c[nd - 2];
    T c1 = c[nd - 1];
    for (int j = nd - 2; j >= 1; --j) {
        T k = T(j) / T(j + 1);
        T temp = c0;
        c0 = c[j - 1] - c1 * k;
        c1 = temp + c1 * x * (k + 1);
    }
    return c0 + c1 * x;
}

template <typename T>
Matrix<T, Dynamic, Dynamic> legvander(const Eigen::VectorX<T> &x, int deg)
{
    if (deg < 0) {
        throw std::domain_error("degree needs to be non-negative");
    }

    int n = x.size();
    Matrix<T, Dynamic, Dynamic> v(n, deg + 1);

    // Use forward recursion to generate the entries
    for (int i = 0; i < n; ++i) {
        v(i, 0) = T(1);
    }
    if (deg > 0) {
        for (int i = 0; i < n; ++i) {
            v(i, 1) = x[i];
        }
        for (int i = 2; i <= deg; ++i) {
            T invi = T(1) / T(i);
            for (int j = 0; j < deg + 1; ++j) {
                v(j, i) =
                    v(j, i - 1) * x[j] * (2 - invi) - v(j, i - 2) * (1 - invi);
            }
        }
    }

    return v;
}

// Add legder for accepting std::vector<T>
template <typename T>
Matrix<T, Dynamic, Dynamic> legvander(const std::vector<T> &x, int deg)
{
    Eigen::VectorX<T> x_eigen =
        Eigen::Map<const Eigen::VectorX<T>>(x.data(), x.size());
    return legvander(x_eigen, deg);
}

template <typename T>
Matrix<T, Dynamic, Dynamic> legder(Matrix<T, Dynamic, Dynamic> c, int cnt = 1)
{
    if (cnt < 0) {
        throw std::domain_error(
            "The order of derivation needs to be non-negative");
    }
    if (cnt == 0) {
        return c;
    }

    int n = c.rows();
    int m = c.cols();
    if (cnt >= n) {
        return Matrix<T, Dynamic, Dynamic>::Zero(1, m);
    }

    for (int k = 0; k < cnt; ++k) {
        n -= 1;
        Matrix<T, Dynamic, Dynamic> der(n, m);
        for (int j = n; j >= 2; --j) {
            der.row(j - 1) = (2 * j - 1) * c.row(j);
            c.row(j - 2) += c.row(j);
        }
        if (n > 1) {
            der.row(1) = 3 * c.row(2);
        }
        der.row(0) = c.row(1);
        c = der;
    }
    return c;
}

// -----------------------------
// Implementation of Bessel function J_mu(x) using series expansion
// -----------------------------
// Definition:
//  J_mu(x) = sum_{m=0}^∞ (-1)^m / (m! Gamma(m+mu+1)) * (x/2)^(2m+mu)
// Note: Using std::tgamma (available in C++11 and later)
inline double _besselj_series(double mu, double x)
{
    const int maxIter = 100;
    double sum = 0.0;
    double halfx = x / 2.0;
    for (int m = 0; m < maxIter; ++m) {
        // (-1)^m
        double sign = (m % 2 == 0) ? 1.0 : -1.0;
        // Numerator: (x/2)^(2*m + mu)
        double termNumerator = std::pow(halfx, 2 * m + mu);
        // Denominator: m! * Gamma(m + mu + 1)
        double termDenom = std::tgamma(m + 1) * std::tgamma(m + mu + 1);
        double term = sign * termNumerator / termDenom;
        sum += term;
        // Convergence check (terminate when relative error is small enough)
        if (std::abs(term) < 1e-14 * std::abs(sum)) {
            break;
        }
    }
    return sum;
}

// Calculation of Bessel function J_mu(x) (using series expansion)
// Note: Precision may decrease for large values of x
inline double _besselj(double mu, double x) { return _besselj_series(mu, x); }

// (2) For general real order
// By definition:
//   j_ν(x) = sqrt(pi/(2x)) * J_{ν+1/2}(x)
inline double sphericalbesselj(int nu, double x)
{
    if (x < 0.0) {
        throw std::domain_error("sphericalbesselj: x must be nonnegative.");
    }
    if (x == 0.0) {
        // For x == 0, j_0(0)=1 when ν=0, otherwise 0 (real solution)
        return (nu == 0.0 ? 1.0 : 0.0);
    }
    double mu = nu + 0.5; // Half-integer exponent
    double J = _besselj(mu, x);
    return std::sqrt(M_PI / (2 * x)) * J;
}

} // namespace sparseir