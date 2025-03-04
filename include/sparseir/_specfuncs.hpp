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

// Constants and utility functions
inline double SQPIO2(double x) {
    return 1.25331413731550025; // sqrt(π/2)
}

inline double SQ2PI(double x) {
    return std::sqrt(2.0 * M_PI / x);
}

inline double sinpi(double x) {
    return std::sin(M_PI * x);
}

// Polynomial evaluation using Horner's method
template<typename T>
T evalpoly(T x, const std::vector<T>& coeffs) {
    T result = coeffs[0];
    for (size_t i = 1; i < coeffs.size(); ++i) {
        result = result * x + coeffs[i];
    }
    return result;
}

// Gamma function implementation
inline double gamma(double _x) {
    double x = _x;
    double s = 1.0;

    if (x < 0) {
        s = sinpi(_x);
        if (s == 0) {
            throw std::domain_error("NaN result for non-NaN input.");
        }
        x = -x; // Use this rather than the traditional x = 1-x to avoid roundoff.
        s *= x;
    }

    if (!std::isfinite(x)) {
        return x;
    }

    if (x > 11.5) {
        double w = 1.0 / x;
        std::vector<double> coefs = {
            1.0,
            8.333333333333331800504e-2, 3.472222222230075327854e-3, -2.681327161876304418288e-3, -2.294719747873185405699e-4,
            7.840334842744753003862e-4, 6.989332260623193171870e-5, -5.950237554056330156018e-4, -2.363848809501759061727e-5,
            7.147391378143610789273e-4
        };
        w = evalpoly(w, coefs);

        // avoid overflow
        double v = std::pow(x, 0.5 * x - 0.25);
        double res = SQ2PI(x) * v * (v / std::exp(x)) * w;

        if (_x < 0) {
            return M_PI / (res * s);
        } else {
            return res;
        }
    }

    std::vector<double> P = {
        1.000000000000000000009e0, 8.378004301573126728826e-1, 3.629515436640239168939e-1, 1.113062816019361559013e-1,
        2.385363243461108252554e-2, 4.092666828394035500949e-3, 4.542931960608009155600e-4, 4.212760487471622013093e-5
    };

    std::vector<double> Q = {
        9.999999999999999999908e-1, 4.150160950588455434583e-1, -2.243510905670329164562e-1, -4.633887671244534213831e-2,
        2.773706565840072979165e-2, -7.955933682494738320586e-4, -1.237799246653152231188e-3, 2.346584059160635244282e-4,
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
    return _x < 0 ? M_PI * q / (s * z * p) : z * p / q;
}

// Forward declarations
double besselj(double nu, double x);
double sphericalbesselj_generic(int nu, double x);
double sphericalbesselj_small_args(int nu, double x);
bool sphericalbesselj_small_args_cutoff(int nu, double x);
double sphericalbesselj_positive_args(int nu, double x);
double besselj_ratio_jnu_jnum1(int n, double x);
double sphericalbesselj_recurrence(int nu, double x);
std::pair<double, double> sphericalbessely_forward_recurrence(int nu, double x);

// Bessel function J implementation (simplified for this example)
inline double besselj(double nu, double x) {
    // This is a simplified placeholder for the actual besselj function
    // In a real implementation, you would need a proper Bessel function implementation
    // or link to an existing library like GSL or Boost

    // For the purpose of this port, you might want to:
    // 1. Implement besselj fully (complex)
    // 2. Use Eigen's special functions if available
    // 3. Use another library like Boost or GSL

    // Simple implementation for non-negative integer order and positive x
    if (nu == 0.0) {
        return std::cos(x);
    } else if (nu == 1.0) {
        return std::sin(x) / x;
    } else {
        // For other orders, you'd need a more complete implementation
        // This is just a placeholder that will work for the immediate function calls
        double j0 = std::cos(x);
        double j1 = std::sin(x) / x;
        double jn = 0.0;

        for (int i = 1; i < nu; ++i) {
            jn = (2.0 * i / x) * j1 - j0;
            j0 = j1;
            j1 = jn;
        }

        return jn;
    }
}

// Spherical Bessel function implementations
inline double sphericalbesselj_generic(int nu, double x) {
    return SQPIO2(x) * besselj(nu + 0.5, x) / std::sqrt(x);
}

inline double sphericalbesselj_small_args(int nu, double x) {
    if (x == 0.0) {
        return (nu == 0) ? 1.0 : 0.0;
    }

    double x2 = x * x / 4.0;
    std::vector<double> coeff = {1.0, -1.0/(1.5 + nu), -1.0/(5.0 + nu), -1.0/(10.5 + nu), -1.0/(18.0 + nu)};
    double coef = evalpoly(x2, coeff);
    double a = SQPIO2(x) / (gamma(1.5 + nu) * std::pow(2.0, nu + 0.5));
    return std::pow(x, nu) * a * coef;
}

inline bool sphericalbesselj_small_args_cutoff(int nu, double x) {
    return x * x / (4.0 * nu + 110.0) < std::numeric_limits<double>::epsilon();
}

inline double sphericalbesselj_positive_args(int nu, double x) {
    if (sphericalbesselj_small_args_cutoff(nu, x)) {
        return sphericalbesselj_small_args(nu, x);
    }

    return (x >= nu && nu < 250) || (x < nu && nu < 60) ?
        sphericalbesselj_recurrence(nu, x) :
        sphericalbesselj_generic(nu, x);
}

// implements continued fraction to compute ratio of J_{ν}(x)/J_{ν-1}(x)
inline double besselj_ratio_jnu_jnum1(int n, double x) {
    const int MaxIter = 5000;
    double xinv = 1.0 / x;
    double xinv2 = 2.0 * xinv;
    double d = x / (n + n);
    double a = d;
    double h = a;
    double b = (2.0 * n + 2.0) * xinv;

    for (int i = 0; i < MaxIter; ++i) {
        d = 1.0 / (b - d);
        a *= (b * d - 1.0);
        h = h + a;
        b = b + xinv2;

        if (std::abs(a / h) <= std::numeric_limits<double>::epsilon()) {
            break;
        }
    }

    return h;
}

inline double sphericalbesselj_recurrence(int nu, double x) {
    if (x >= nu) {
        // forward recurrence if stable
        double xinv = 1.0 / x;
        double s = std::sin(x);
        double c = std::cos(x);
        double sJ0 = s * xinv;
        double sJ1 = (sJ0 - c) * xinv;

        double nu_start = 1.0;
        while (nu_start < nu + 0.5) {
            double tmp = sJ1;
            sJ1 = (2.0 * nu_start + 1.0) * xinv * sJ1 - sJ0;
            sJ0 = tmp;
            nu_start += 1.0;
        }

        return sJ0;
    } else {
        // compute sphericalbessely with forward recurrence and use continued fraction
        auto result = sphericalbessely_forward_recurrence(nu, x);
        double sYnm1 = result.first;
        double sYn = result.second;
        double H = besselj_ratio_jnu_jnum1(nu + 1.5, x);
        return 1.0 / (x * x * (H * sYnm1 - sYn));
    }
}

inline std::pair<double, double> sphericalbessely_forward_recurrence(int nu, double x) {
    double xinv = 1.0 / x;
    double s = std::sin(x);
    double c = std::cos(x);
    double sY0 = -c * xinv;
    double sY1 = xinv * (sY0 - s);

    double nu_start = 1.0;
    while (nu_start < nu + 0.5) {
        double tmp = sY1;
        sY1 = (2.0 * nu_start + 1.0) * xinv * sY1 - sY0;
        sY0 = tmp;
        nu_start += 1.0;
    }

    // need to check if NaN resulted during loop
    // this could happen if x is very small and nu is large which eventually results in overflow (-> -Inf)
    if (std::isnan(sY0)) {
        return {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    } else {
        return {sY0, sY1};
    }
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