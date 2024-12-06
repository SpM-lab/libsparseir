
#pragma once

#include <vector>

namespace sparseir{
/**
 * @file abstract_sve.hpp
 * @brief Abstract base class for Singular Value Expansion (SVE) strategies.
 *
 * This class corresponds to the AbstractSVE in Julia's abstract.jl.
 */

class AbstractSVE
{
public:
    virtual ~AbstractSVE() = default;

    /**
     * Compute the matrices underlying the SVE.
     *
     * This method should be overridden by derived classes to return the matrices
     * that will be used for SVD computation.
     */
    virtual std::vector<Eigen::MatrixXd> matrices() const = 0;

    // You can add any common methods or virtual interfaces here that are shared among SVE classes.
};

template <typename K, typename T=double>
class SamplingSVE : public AbstractSVE
{
public:
    K kernel;
    double epsilon;
    int n_gauss;
    int nsvals_hint;

    Rule<T> rule;
    std::vector<T> segs_x;
    std::vector<T> segs_y;
    Rule<T> gauss_x;
    Rule<T> gauss_y;

    // Constructor
    SamplingSVE(const K& kernel_, double epsilon_, int n_gauss_ = -1)
        : kernel(kernel_), epsilon(epsilon_), n_gauss(n_gauss_)
    {
        // sve_hints_ = sve_hints(kernel, ε)
        auto sve_hints_ = sve_hints(kernel, epsilon);

        // n_gauss = something(n_gauss, ngauss(sve_hints_))
        if (n_gauss <= 0)
            n_gauss = ngauss(sve_hints_);

        // rule = legendre(n_gauss, T)

        rule = legendre(n_gauss);

        // segs_x, segs_y = segments_x(sve_hints_, T), segments_y(sve_hints_, T)
        segs_x = segments_x(sve_hints_);
        segs_y = segments_y(sve_hints_);

        // gauss_x, gauss_y = piecewise(rule, segs_x), piecewise(rule, segs_y)
        gauss_x = piecewise(rule, segs_x);
        gauss_y = piecewise(rule, segs_y);

        nsvals_hint = nsvals(sve_hints_);
    }
};

template <typename K, typename T, template<typename, typename> class InnerSVE = SamplingSVE>
class CentrosymmSVE : public AbstractSVE
{
public:
    K kernel;
    double epsilon;
    InnerSVE<T, K> even;
    InnerSVE<T, K> odd;
    int nsvals_hint;

    CentrosymmSVE(const K& kernel_, double epsilon_, int n_gauss_ = -1)
        : kernel(kernel_), epsilon(epsilon_)
    {
        // even = InnerSVE(get_symmetrized(kernel, +1), ε, T; n_gauss)
        auto kernel_even = get_symmetrized(kernel_, +1);
        even = InnerSVE<T, K>(kernel_even, epsilon_, n_gauss_);

        // odd = InnerSVE(get_symmetrized(kernel, -1), ε, T; n_gauss)
        auto kernel_odd = get_symmetrized(kernel_, -1);
        odd = InnerSVE<T, K>(kernel_odd, epsilon_, n_gauss_);

        nsvals_hint = std::max(even.nsvals_hint, odd.nsvals_hint);
    }
};

} // namespace sparseir

