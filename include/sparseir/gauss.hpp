#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <xprec/ddouble-header-only.hpp>

namespace sparseir {

using namespace Eigen;
using xprec::DDouble;

class slice {
public:
    size_t start, stop, step;

    slice(size_t start, size_t stop, size_t step = 1)
        : start(start), stop(stop), step(step) {}

    std::vector<size_t> indices() const {
        std::vector<size_t> result;
        for (size_t i = start; i < stop; i += step) {
            result.push_back(i);
        }
        return result;
    }
};

/*
Quadrature rule.

    Approximation of an integral by a weighted sum over discrete points:

         ∫ f(x) * omega(x) * dx ~ sum(f(xi) * wi for (xi, wi) in zip(x, w))

    where we generally have superexponential convergence for smooth ``f(x)``
    with the number of quadrature points.
*/

template <typename T>
class Rule {
public:
    // TODO: Define x and w as Eigen::VectorXd
    std::vector<T> x, w, x_forward, x_backward;
    T a, b;
    Rule(const std::vector<T>& x, const std::vector<T>& w, std::vector<T> x_forward, std::vector<T> x_backward, T a = -1, T b = 1)
        : x(x), w(w), x_forward(x_forward), x_backward(x_backward), a(a), b(b) {}
    Rule(const std::vector<T>& x, const std::vector<T>& w, T a = -1, T b = 1)
        : x(x), w(w), a(a), b(b) {
        this->x_forward = x_forward.empty() ? std::vector<T>(x.size(), 0) : x_forward;
        this->x_backward = x_backward.empty() ? std::vector<T>(x.size(), 0) : x_backward;
        if (x_forward.empty()) {
            transform(x.begin(), x.end(), this->x_forward.begin(), [a](T xi) { return xi - a; });
        }
        if (x_backward.empty()) {
            transform(x.begin(), x.end(), this->x_backward.begin(), [b](T xi) { return b - xi; });
        }
    }

    Rule<T> reseat(T a, T b) const {
        T scaling = (b - a) / (this->b - this->a);
        std::vector<T> new_x(x.size()), new_w(w.size()), new_x_forward(x_forward.size()), new_x_backward(x_backward.size());
        transform(x.begin(), x.end(), new_x.begin(), [this, scaling, a, b](T xi) { return scaling * (xi - (this->b + this->a) / 2) + (b + a) / 2; });
        transform(w.begin(), w.end(), new_w.begin(), [scaling](T wi) { return wi * scaling; });
        transform(x_forward.begin(), x_forward.end(), new_x_forward.begin(), [scaling](T xi) { return xi * scaling; });
        transform(x_backward.begin(), x_backward.end(), new_x_backward.begin(), [scaling](T xi) { return xi * scaling; });
        return Rule<T>(new_x, new_w, new_x_forward, new_x_backward, a, b);
    }

    Rule<T> scale(T factor) const {
        std::vector<T> new_w(w.size());
        transform(w.begin(), w.end(), new_w.begin(), [factor](T wi) { return wi * factor; });
        return Rule<T>(x, new_w, x_forward, x_backward, a, b);
    }

    template <typename U>
    Rule<T> piecewise(const std::vector<U>& edges) const {
        if (!std::is_sorted(edges.begin(), edges.end())) {
            throw std::invalid_argument("segments ends must be ordered ascendingly");
        }
        std::vector<Rule<T>> rules;
        for (size_t i = 0; i < edges.size() - 1; ++i) {
            rules.push_back(reseat(T(edges[i]), T(edges[i + 1])));
        }
        return join(rules);
    }

    Rule<T> astype(const std::string& dtype) const {
        // Assuming dtype is either "float" or "double"
        return *this;
    }

    static Rule<T> join(const std::vector<Rule<T>>& gauss_list) {
        if (gauss_list.empty()) {
            return Rule<T>({}, {});
        }

        T a = gauss_list.front().a;
        T b = gauss_list.back().b;
        T prev_b = a;
        std::vector<T> x, w, x_forward, x_backward;

        for (const auto& curr : gauss_list) {
            if (curr.a != prev_b) {
                throw std::invalid_argument("Gauss rules must be ascending");
            }
            prev_b = curr.b;
            std::vector<T> curr_x_forward(curr.x_forward.size()), curr_x_backward(curr.x_backward.size());
            transform(curr.x_forward.begin(), curr.x_forward.end(), curr_x_forward.begin(), [a, curr](T xi) { return xi + (curr.a - a); });
            transform(curr.x_backward.begin(), curr.x_backward.end(), curr_x_backward.begin(), [b, curr](T xi) { return xi + (b - curr.b); });
            x.insert(x.end(), curr.x.begin(), curr.x.end());
            w.insert(w.end(), curr.w.begin(), curr.w.end());
            x_forward.insert(x_forward.end(), curr_x_forward.begin(), curr_x_forward.end());
            x_backward.insert(x_backward.end(), curr_x_backward.begin(), curr_x_backward.end());
        }

        return Rule<T>(x, w, x_forward, x_backward, a, b);
    }
};

/*
template <typename T>
class NestedRule : public Rule<T> {
public:
    std::vector<T> v;
    slice vsel;

    NestedRule(const std::vector<T>& x, const std::vector<T>& w, const std::vector<T>& v, const std::vector<T>& x_forward = {}, const std::vector<T>& x_backward = {}, T a = -1, T b = 1)
        : Rule<T>(x, w, x_forward, x_backward, a, b), v(v), vsel(1, v.size(), 2) {}

    NestedRule<T> reseat(T a, T b) const {
        Rule<T> res = Rule<T>::reseat(a, b);
        std::vector<T> new_v(v.size());
        transform(v.begin(), v.end(), new_v.begin(), [this, a, b](T vi) { return (b - a) / (this->b - this->a) * vi; });
        return NestedRule<T>(res.x, res.w, new_v, res.x_forward, res.x_backward, res.a, res.b);
    }

    NestedRule<T> scale(T factor) const {
        Rule<T> res = Rule<T>::scale(factor);
        std::vector<T> new_v(v.size());
        transform(v.begin(), v.end(), new_v.begin(), [factor](T vi) { return vi * factor; });
        return NestedRule<T>(res.x, res.w, new_v, res.x_forward, res.x_backward, res.a, res.b);
    }

    NestedRule<T> astype(const string& dtype) const {
        // Assuming dtype is either "float" or "double"
        return *this;
    }
};
*/

/*
    legendre(n[, T])

Gauss-Legendre quadrature with `n` points on [-1, 1].
*/
inline Rule<DDouble> legendre(int n){
    std::vector<DDouble> x(n), w(n);
    xprec::gauss_legendre(n, x.data(), w.data());
    return Rule<DDouble>(x, w);
}


template <typename T>
Matrix<T, Dynamic, Dynamic> legendre_collocation(const Rule<T>& rule, int n = -1) {
    if (n < 0) {
        n = rule.x.size();
    }
    // Compute the Legendre Vandermonde matrix
    Matrix<T, Dynamic, Dynamic> lv = legvander(rule.x, n-1);
    for (size_t i = 0; i < rule.w.size(); ++i) {
        for (size_t j = 0; j < lv.cols(); ++j) {
            lv(i, j) *= rule.w[i];
        }
    }
    // !!! do NOT do this !!!
    // res = res.transpose();
    // This is the so-called aliasing issue. In "debug mode", i.e., when assertions have not been disabled, such common pitfalls are automatically detected.

    auto res = lv.transpose();

    // Normalize the matrix rows
    VectorXd invnorm = VectorXd::LinSpaced(n, 0.5, n - 0.5);

    for (size_t i = 0; i < invnorm.size(); ++i) {
        for (size_t j = 0; j < lv.cols(); ++j) {
            res(i, j) *= invnorm(i);
        }
    }

    return res;
}

template <typename TargetType, typename SourceType>
inline Rule<TargetType> convert(const Rule<SourceType> &rule)
{
    std::vector<TargetType> x(rule.x.begin(), rule.x.end());
    std::vector<TargetType> w(rule.w.begin(), rule.w.end());
    TargetType a = static_cast<TargetType>(rule.a);
    TargetType b = static_cast<TargetType>(rule.b);
    std::vector<TargetType> x_forward(rule.x_forward.begin(), rule.x_forward.end());
    std::vector<TargetType> x_backward(rule.x_backward.begin(), rule.x_backward.end());

    return Rule<TargetType>(x, w, x_forward, x_backward, a, b);
}

} // namespace sparseir