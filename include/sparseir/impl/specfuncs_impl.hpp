#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>

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

// Constants
const double PI = 3.14159265358979323846;

inline double SQPIO2(double /*x*/) {
    // Returns sqrt(pi/2) (argument unused)
    return 1.25331413731550025;
}

// Helper: evaluate polynomial using Horner's method.
inline double evalpoly(double x, const std::vector<double>& coeffs) {
    double result = 0.0;
    for (auto it = coeffs.rbegin(); it != coeffs.rend(); ++it)
        result = result * x + *it;
    return result;
}

// Helper: compute sin(pi*x)
inline double sinpi(double x) {
    return std::sin(PI * x);
}

// Gamma function approximation (for real x)
inline double gamma_func(double _x) {
    double x = _x;
    double s = 0.0;
    if (x < 0.0) {
        s = sinpi(_x);
        if (s == 0.0)
            throw std::domain_error("NaN result for non-NaN input.");
        x = -x; // Use this rather than 1-x to avoid roundoff.
        s *= x;
    }
    if (!std::isfinite(x))
        return x;

    if (x > 11.5) {
        double w = 1.0 / x;
        std::vector<double> coefs = {
            1.0,
            8.333333333333331800504e-2,
            3.472222222230075327854e-3,
            -2.681327161876304418288e-3,
            -2.294719747873185405699e-4,
            7.840334842744753003862e-4,
            6.989332260623193171870e-5,
            -5.950237554056330156018e-4,
            -2.363848809501759061727e-5,
            7.147391378143610789273e-4
        };
        w = evalpoly(w, coefs);
        // v = x^(0.5*x - 0.25)
        double v = std::pow(x, 0.5 * x - 0.25);
        double res = SQPIO2(x) * v * (v / std::exp(x)) * w;
        return (_x < 0.0) ? (PI / (res * s)) : res;
    }

    std::vector<double> P = {
        1.000000000000000000009e0, 8.378004301573126728826e-1,
        3.629515436640239168939e-1, 1.113062816019361559013e-1,
        2.385363243461108252554e-2, 4.092666828394035500949e-3,
        4.542931960608009155600e-4, 4.212760487471622013093e-5
    };
    std::vector<double> Q = {
        9.999999999999999999908e-1, 4.150160950588455434583e-1,
        -2.243510905670329164562e-1, -4.633887671244534213831e-2,
        2.773706565840072979165e-2, -7.955933682494738320586e-4,
        -1.237799246653152231188e-3, 2.346584059160635244282e-4,
        -1.397148517476170440917e-5
    };

    double z = 1.0;
    while (x >= 3.0) {
        x -= 1.0;
        z *= x;
    }
    while (x < 2.0) {
        z /= x;
        x += 1.0;
    }
    x -= 2.0;
    double p = evalpoly(x, P);
    double q = evalpoly(x, Q);
    return (_x < 0.0) ? (PI * q / (s * z * p)) : (z * p / q);
}

// Cylindrical Bessel function of the first kind, J_nu(x)
// Uses the series expansion:
//   J_nu(x) = sum_{m=0}^∞ (-1)^m / (m! * Gamma(nu+m+1)) * (x/2)^(2m+nu)
inline double cyl_bessel_j(double nu, double x) {
    const int maxIter = 1000;
    const double eps = std::numeric_limits<double>::epsilon();
    double sum = 0.0;
    double term = std::pow(x / 2.0, nu) / gamma_func(nu + 1.0);
    sum = term;
    for (int m = 1; m < maxIter; m++) {
        term *= - (x * x / 4.0) / (m * (nu + m));
        sum += term;
        if (std::abs(term) < std::abs(sum) * eps)
            break;
    }
    return sum;
}

// sphericalbesselj_generic:
// Computes the spherical Bessel function j_n(x) using the relation:
//   j_n(x) = sqrt(pi/(2x)) * J_{n+1/2}(x)
inline double sphericalbesselj_generic(double nu, double x) {
    return SQPIO2(x) * cyl_bessel_j(nu + 0.5, x) / std::sqrt(x);
}

// sphericalbesselj_small_args:
// Approximation for small x.
inline double sphericalbesselj_small_args(double nu, double x) {
    if (x == 0.0)
        return (nu == 0.0) ? 1.0 : 0.0;
    double x2 = (x * x) / 4.0;
    std::vector<double> coef = {
        1.0,
        -1.0 / (1.5 + nu),   // 3/2 + nu
        -1.0 / (5.0 + nu),
        -1.0 / ((21.0 / 2.0) + nu), // 21/2 + nu
        -1.0 / (18.0 + nu)
    };
    double a = SQPIO2(x) / (gamma_func(1.5 + nu) * std::pow(2.0, nu + 0.5));
    return std::pow(x, nu) * a * evalpoly(x2, coef);
}

// sphericalbesselj_small_args_cutoff:
// Determines when the small-argument expansion is accurate.
inline bool sphericalbesselj_small_args_cutoff(double nu, double x) {
    return ((x * x) / (4 * nu + 110)) < std::numeric_limits<double>::epsilon();
}

// besselj_ratio_jnu_jnum1:
// Computes the continued-fraction for the ratio J_{ν}(x) / J_{ν-1}(x)
inline double besselj_ratio_jnu_jnum1(double n, double x) {
    const int MaxIter = 5000;
    double xinv = 1.0 / x;
    double xinv2 = 2.0 * xinv;
    double d = x / (2.0 * n);
    double a = d;
    double h = a;
    double b = (2.0 * n + 2.0) * xinv;
    for (int i = 0; i < MaxIter; i++) {
        d = 1.0 / (b - d);
        a *= (b * d - 1.0);
        h += a;
        b += xinv2;
        if (std::abs(a / h) <= std::numeric_limits<double>::epsilon())
            break;
    }
    return h;
}

// sphericalbessely_forward_recurrence:
// Computes forward recurrence for spherical Bessel y.
// Returns a pair: (sY_{n-1}, sY_n)
inline std::pair<double, double> sphericalbessely_forward_recurrence(int nu, double x) {
    double xinv = 1.0 / x;
    double s = std::sin(x);
    double c = std::cos(x);
    double sY0 = -c * xinv;
    double sY1 = xinv * (sY0 - s);
    double nu_start = 1.0;
    while (nu_start < nu + 0.5) {
        double temp = sY1;
        sY1 = ((2.0 * nu_start + 1.0) * xinv * sY1 - sY0);
        sY0 = temp;
        nu_start += 1.0;
    }
    if (std::isnan(sY0))
        return std::make_pair(-std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity());
    return std::make_pair(sY0, sY1);
}

// sphericalbesselj_recurrence:
// Uses forward recurrence if stable; otherwise uses spherical Bessel y recurrence.
inline double sphericalbesselj_recurrence(int nu, double x) {
    if (x >= nu) {
        double xinv = 1.0 / x;
        double s = std::sin(x);
        double c = std::cos(x);
        double sJ0 = s * xinv;
        double sJ1 = (sJ0 - c) * xinv;
        double nu_start = 1.0;
        while (nu_start < nu + 0.5) {
            double temp = sJ1;
            sJ1 = ((2.0 * nu_start + 1.0) * xinv * sJ1 - sJ0);
            sJ0 = temp;
            nu_start += 1.0;
        }
        return sJ0;
    } else {
        std::pair<double, double> y_pair = sphericalbessely_forward_recurrence(nu, x);
        double sYnm1 = y_pair.first;
        double sYn = y_pair.second;
        double H = besselj_ratio_jnu_jnum1(nu + 1.5, x);
        return 1.0 / (x * x * (H * sYnm1 - sYn));
    }
}

// sphericalbesselj_positive_args:
// Selects the proper method for computing j_n(x) for positive arguments.
inline double sphericalbesselj_positive_args(int nu, double x) {
    if (sphericalbesselj_small_args_cutoff(nu, x))
        return sphericalbesselj_small_args(nu, x);
    if ((x >= nu && nu < 250) || (x < nu && nu < 60))
        return sphericalbesselj_recurrence(nu, x);
    else
        return sphericalbesselj_generic(nu, x);
}


// Main function to calculate spherical Bessel function of the first kind
inline double sphericalbesselj(int n, double x) {
    // Handle negative arguments
    if (x < 0.0) {
        throw std::domain_error("sphericalBesselJ requires non-negative x");
    }
    return sphericalbesselj_positive_args(n, x);
}

} // namespace sparseir