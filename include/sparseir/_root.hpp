#pragma once

#include <algorithm>
#include <bitset>
#include <cmath>
#include <functional>
#include <vector>

namespace sparseir {

template <typename T>
T midpoint(T lo, T hi)
{
    if (std::is_integral<T>::value) {
        return lo + ((hi - lo) >> 1);
    } else {
        return lo + ((hi - lo) * static_cast<T>(0.5));
    }
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
        x_bisect.push_back(bisect(f, a[i], b[i], fa[i], epsilon_x));
    }

    x_hit.insert(x_hit.end(), x_bisect.begin(), x_bisect.end());
    std::sort(x_hit.begin(), x_hit.end());
    return x_hit;
}

template <typename F, typename T>
T bisect(F f, T a, T b, double fa, double epsilon_x)
{
    while (true) {
        T mid = midpoint(a, b);
        if (closeenough(a, mid, epsilon_x))
            return mid;
        double fmid = f(mid);
        if (std::signbit(fa) != std::signbit(fmid)) {
            b = mid;
        } else {
            a = mid;
            fa = fmid;
        }
    }
}

template <typename T>
bool closeenough(T a, T b, double epsilon)
{
    if (std::is_floating_point<T>::value) {
        return std::abs(a - b) <= epsilon;
    } else {
        return a == b;
    }
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