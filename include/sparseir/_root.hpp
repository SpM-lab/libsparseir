#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

#include <iostream>

#include <Eigen/Dense>

namespace sparseir {

// Midpoint function for floating-point types
template<typename T1, typename T2>
typename std::enable_if<
    std::is_floating_point<T1>::value && std::is_floating_point<T2>::value,
    typename std::common_type<T1, T2>::type
>::type
inline midpoint(T1 a, T2 b) {
    typedef typename std::common_type<T1, T2>::type CommonType;
    return static_cast<CommonType>(a) +
           (static_cast<CommonType>(b) - static_cast<CommonType>(a)) *
           static_cast<CommonType>(0.5);
}

// Midpoint function for integral types
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
inline midpoint(T a, T b) {
    return a + ((b - a) / 2);
}

// Close enough function for floating-point types
template<typename T>
inline bool closeenough(T a, T b, T epsilon) {
    return std::abs(a - b) <= epsilon;
}

template<typename T>
inline bool closeenough(int a, int b, T _dummyepsilon) {
    return a == b;
}

// Signbit function (handles both floating-point and integral types)
template<typename T>
inline bool signbit(T x) {
    return x < static_cast<T>(0);
}

// Bisection method to find a root of function f in [a, b]
template<typename F>
double bisect(F f, double a, double b, double fa, double epsilon_x) {
    //while (true) {
    double mid = midpoint(a, b);
    double fmid = f(mid);
    for (int dummy = 0; dummy < 100; ++dummy) {
        double mid = midpoint(a, b);
        if (closeenough(a, mid, epsilon_x)) {
            return mid;
        }
        fmid = f(mid);
        std::cout << "a = " << a << ", b = " << b << ", mid = " << mid << ", fmid = " << fmid << ", epsilon_x = " << epsilon_x << std::endl;
        std::cout << "a, mid, epsilon_x = " << a << ", " << mid << ", " << epsilon_x << std::endl;
        std::cout << "std::abs(a - mid) = " << std::abs(a - mid) << std::endl;
        std::cout << "closeenough(a, mid, epsilon_x) = " << closeenough(a, mid, epsilon_x) << std::endl;
        if (signbit(fa) != signbit(fmid)) {
            b = mid;
        } else {
            a = mid;
            fa = fmid;
        }
    }
    return fmid;
}

template <typename F, typename T>
std::vector<T> find_all(F f, const std::vector<T> &xgrid)
{
    std::vector<double> fx;
    std::transform(xgrid.begin(), xgrid.end(), std::back_inserter(fx),
                   [&](T x) { return static_cast<double>(f(x)); });

    std::vector<bool> hit(fx.size());
    std::transform(fx.begin(), fx.end(), hit.begin(),
                   [](double val) { return val == 0.0; });

    std::vector<T> x_hit;
    for (size_t i = 0; i < hit.size(); ++i) {
        if (hit[i])
            x_hit.push_back(xgrid[i]);
    }

    std::vector<bool> sign_change(fx.size() - 1);
    for (size_t i = 0; i < fx.size() - 1; ++i) {
        sign_change[i] = std::signbit(fx[i]) != std::signbit(fx[i + 1]);
    }

    for (size_t i = 0; i < sign_change.size(); ++i) {
        sign_change[i] = sign_change[i] && !hit[i] && !hit[i + 1];
    }

    if (std::none_of(sign_change.begin(), sign_change.end(),
                     [](bool v) { return v; })) {
        return x_hit;
    }

    std::vector<bool> where_a(sign_change.size() + 1, false);
    std::vector<bool> where_b(sign_change.size() + 1, false);
    for (size_t i = 0; i < sign_change.size(); ++i) {
        where_a[i] = sign_change[i];
        where_b[i + 1] = sign_change[i];
    }

    std::vector<T> a, b;
    std::vector<double> fa;
    for (size_t i = 0; i < where_a.size(); ++i) {
        if (where_a[i]) {
            a.push_back(xgrid[i]);
            fa.push_back(fx[i]);
        }
        if (where_b[i]) {
            b.push_back(xgrid[i]);
        }
    }

    double epsilon_x =
        std::numeric_limits<T>::epsilon() *
        *std::max_element(xgrid.begin(), xgrid.end(),
                          [](T a, T b) { return std::abs(a) < std::abs(b); });

    std::vector<T> x_bisect;
    for (size_t i = 0; i < a.size(); ++i) {
        double root = bisect(f, a[i], b[i], fa[i], epsilon_x);
        x_bisect.push_back(static_cast<T>(root));
    }

    x_hit.insert(x_hit.end(), x_bisect.begin(), x_bisect.end());
    std::sort(x_hit.begin(), x_hit.end());
    return x_hit;
}

template <typename T>
std::vector<T> refine_grid(const std::vector<T> &grid, int alpha)
{
    size_t n = grid.size();
    size_t newn = alpha * (n - 1) + 1;
    std::vector<T> newgrid(newn);

    for (size_t i = 0; i < n - 1; ++i) {
        T xb = grid[i];
        T xe = grid[i + 1];
        T delta_x = (xe - xb) / alpha;
        size_t newi = alpha * i;
        for (int j = 0; j < alpha; ++j) {
            newgrid[newi + j] = xb + delta_x * j;
        }
    }
    newgrid.back() = grid.back();
    return newgrid;
}

template <typename F, typename T>
T bisect_discr_extremum(F absf, T a, T b, double absf_a, double absf_b)
{
    T d = b - a;

    if (d <= 1)
        return absf_a > absf_b ? a : b;
    if (d == 2)
        return a + 1;

    T m = midpoint(a, b);
    T n = m + 1;
    double absf_m = absf(m);
    double absf_n = absf(n);

    if (absf_m > absf_n) {
        return bisect_discr_extremum(absf, a, n, absf_a, absf_n);
    } else {
        return bisect_discr_extremum(absf, m, b, absf_m, absf_b);
    }
}

template <typename F, typename T>
std::vector<T> discrete_extrema(F f, const std::vector<T> &xgrid)
{
    std::vector<double> fx(xgrid.size());
    std::transform(xgrid.begin(), xgrid.end(), fx.begin(), f);

    std::vector<double> absfx(fx.size());
    std::transform(fx.begin(), fx.end(), absfx.begin(),
                   [](double val) { return std::abs(val); });

    std::vector<bool> signdfdx(fx.size() - 1);
    for (size_t i = 0; i < fx.size() - 1; ++i) {
        signdfdx[i] = std::signbit(fx[i]) != std::signbit(fx[i + 1]);
    }

    std::vector<bool> derivativesignchange(signdfdx.size() - 1);
    for (size_t i = 0; i < signdfdx.size() - 1; ++i) {
        derivativesignchange[i] = signdfdx[i] != signdfdx[i + 1];
    }

    std::vector<bool> derivativesignchange_a(derivativesignchange.size() + 2,
                                             false);
    std::vector<bool> derivativesignchange_b(derivativesignchange.size() + 2,
                                             false);
    for (size_t i = 0; i < derivativesignchange.size(); ++i) {
        derivativesignchange_a[i] = derivativesignchange[i];
        derivativesignchange_b[i + 2] = derivativesignchange[i];
    }

    std::vector<T> a, b;
    std::vector<double> absf_a, absf_b;
    for (size_t i = 0; i < derivativesignchange_a.size(); ++i) {
        if (derivativesignchange_a[i]) {
            a.push_back(xgrid[i]);
            absf_a.push_back(absfx[i]);
        }
        if (derivativesignchange_b[i]) {
            b.push_back(xgrid[i]);
            absf_b.push_back(absfx[i]);
        }
    }

    std::vector<T> res;
    for (size_t i = 0; i < a.size(); ++i) {
        res.push_back(
            bisect_discr_extremum(f, a[i], b[i], absf_a[i], absf_b[i]));
    }

    std::vector<bool> sfx(fx.size());
    std::transform(fx.begin(), fx.end(), sfx.begin(),
                   [](double val) { return std::signbit(val); });

    if (absfx.front() > absfx[1] || sfx.front() != sfx[1]) {
        res.insert(res.begin(), xgrid.front());
    }
    if (absfx.back() > absfx[absfx.size() - 2] ||
        sfx.back() != sfx[sfx.size() - 2]) {
        res.push_back(xgrid.back());
    }

    return res;
}

} // namespace sparseir