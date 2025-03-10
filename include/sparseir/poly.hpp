#pragma once
// C++ Standard Library headers
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iterator>
// Eigen headers
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace sparseir {

class PiecewiseLegendrePoly {
public:
    int polyorder;
    double xmin;
    double xmax;

    Eigen::VectorXd knots;
    Eigen::VectorXd delta_x;
    Eigen::MatrixXd data;
    int symm;
    int l;

    Eigen::VectorXd xm;
    Eigen::VectorXd inv_xs;
    Eigen::VectorXd norms;

    // Default constructor
    PiecewiseLegendrePoly() = default;

    // Constructor for full initialization
    PiecewiseLegendrePoly(int polyorder,
                          double xmin,
                          double xmax,
                          const Eigen::VectorXd& knots,
                          const Eigen::VectorXd& delta_x,
                          const Eigen::MatrixXd& data,
                          int symm,
                          int l,
                          const Eigen::VectorXd& xm,
                          const Eigen::VectorXd& inv_xs,
                          const Eigen::VectorXd& norms)
        : polyorder(polyorder),
          xmin(xmin),
          xmax(xmax),
          knots(knots),
          delta_x(delta_x),
          data(data),
          symm(symm),
          l(l),
          xm(xm),
          inv_xs(inv_xs),
          norms(norms) {

        if (!data.allFinite()) {
            throw std::invalid_argument("data contains NaN or Inf");
        }

        if (!std::is_sorted(knots.data(), knots.data() + knots.size())) {
            throw std::invalid_argument("knots must be monotonically increasing");
        }

        for (int i = 0; i < delta_x.size(); ++i) {
            if (std::abs(delta_x[i] - (knots[i + 1] - knots[i])) > 1e-10) {
                throw std::invalid_argument("delta_x must work with knots");
            }
        }
    }

    // Constructor for copying existing PiecewiseLegendrePoly with new data and symmetry
    PiecewiseLegendrePoly(const Eigen::MatrixXd& new_data,
                          const PiecewiseLegendrePoly& p)
        : polyorder(p.polyorder),
          xmin(p.xmin),
          xmax(p.xmax),
          knots(p.knots),
          delta_x(p.delta_x),
          data(new_data),
          symm(p.symm),
          l(p.l),
          xm(p.xm),
          inv_xs(p.inv_xs),
          norms(p.norms) {}

    // Constructor used in the deriv function
    PiecewiseLegendrePoly(const Eigen::MatrixXd& new_data,
                          const PiecewiseLegendrePoly& p,
                          int new_symm)
        : polyorder(p.polyorder),
          xmin(p.xmin),
          xmax(p.xmax),
          knots(p.knots),
          delta_x(p.delta_x),
          data(new_data),
          symm(new_symm),
          l(p.l),
          xm(p.xm),
          inv_xs(p.inv_xs),
          norms(p.norms) {}

    // Constructor for building from data and knots with optional delta_x and symmetry
    PiecewiseLegendrePoly(const Eigen::MatrixXd& data,
                          const Eigen::VectorXd& knots,
                          int l,
                          const Eigen::VectorXd& delta_x_ = Eigen::VectorXd(),
                          int symm = 0)
        : knots(knots),
          data(data),
          symm(symm),
          l(l) {
        this->polyorder = data.rows();
        int nsegments = data.cols();

        if (knots.size() != nsegments + 1) {
            throw std::invalid_argument("Invalid knots array");
        }

        this->delta_x = delta_x_.size() > 0 ? delta_x_ : diff(knots);
        this->xm = Eigen::VectorXd::Zero(nsegments);
        for (int i = 0; i < nsegments; ++i) {
            this->xm[i] = 0.5 * (knots[i] + knots[i + 1]);
        }
        this->inv_xs = 2.0 / this->delta_x.array();
        this->norms = this->inv_xs.array().sqrt();

        this->xmin = knots(0);
        this->xmax = knots(knots.size() - 1);
    }

    // Factory method for creating a new PiecewiseLegendrePoly instance
    static PiecewiseLegendrePoly create(const Eigen::MatrixXd& data,
                                        const Eigen::VectorXd& knots,
                                        int l,
                                        const Eigen::VectorXd& delta_x_ = Eigen::VectorXd(),
                                        int symm = 0) {
        return PiecewiseLegendrePoly(data, knots, l, delta_x_, symm);
    }

    // Function call operator: evaluate the polynomial at x
    double operator()(double x) const
    {
        int i;
        double x_tilde;
        std::tie(i, x_tilde) = split(x);
        Eigen::VectorXd coeffs = data.col(i);
        // convert coeffs to std::vector<double>
        std::vector<double> coeffs_vec(coeffs.data(), coeffs.data() + coeffs.size());
        double value = legval<double>(x_tilde, coeffs_vec) * norms(i);
        return value;
    }

    // Evaluate the polynomial at an array of x
    Eigen::VectorXd operator()(const Eigen::VectorXd &xs) const
    {
        Eigen::VectorXd results(xs.size());
        for (int idx = 0; idx < xs.size(); ++idx) {
            results[idx] = (*this)(xs[idx]);
        }
        return results;
    }

    // Overlap function
    double overlap(std::function<double(double)> f,
                   double rtol = std::numeric_limits<double>::epsilon(),
                   bool return_error = false, int maxevals = 10000,
                   const std::vector<double> &points = {}) const
    {
        // Implement numerical integration over the intervals
        // Since C++ does not have a built-in quadgk, we need to implement one
        // or use a library For simplicity, let's use Gauss-Legendre quadrature
        // over each segment
        double result = 0.0;

        std::vector<double> integration_points(knots.data(),
                                               knots.data() + knots.size());
        integration_points.insert(integration_points.end(), points.begin(),
                                  points.end());
        std::sort(integration_points.begin(), integration_points.end());
        integration_points.erase(
            std::unique(integration_points.begin(), integration_points.end()),
            integration_points.end());

        for (size_t idx = 0; idx < integration_points.size() - 1; ++idx) {
            double a = integration_points[idx];
            double b = integration_points[idx + 1];

            // Perform Gauss-Legendre quadrature on [a, b]
            auto integrand = [this, &f](double x) { return (*this)(x)*f(x); };
            double integral = gauss_legendre_quadrature(a, b, integrand);
            result += integral;
        }

        if (return_error) {
            // Since we haven't computed the error estimate, we return zero
            return 0.0; // Placeholder for error estimate
        } else {
            return result;
        }
    }

    // Derivative function
    PiecewiseLegendrePoly deriv(int n = 1) const
    {
        Eigen::MatrixXd ddata = legder(data, n);

        // Multiply each column by inv_xs[i]^n
        for (int i = 0; i < ddata.cols(); ++i) {
            ddata.col(i) *= std::pow(inv_xs[i], n);
        }

        int new_symm = std::pow(-1, n) * symm;
        return PiecewiseLegendrePoly(ddata, *this, new_symm);
    }

    // Function to compute derivatives at a point x
    Eigen::VectorXd derivs(double x) const {
        std::vector<double> res;
        res.push_back((*this)(x)); // Assuming operator() is overloaded for evaluation
        PiecewiseLegendrePoly newppoly = *this;
        for (int i = 2; i <= polyorder; ++i) {
            newppoly = newppoly.deriv();
            res.push_back(newppoly(x));
        }
        //convert to Eigen::VectorXd
        return Eigen::Map<Eigen::VectorXd>(res.data(), res.size());
    }

    Eigen::VectorXd refine_grid(const Eigen::VectorXd& grid, int alpha) const {
        Eigen::VectorXd refined((grid.size() - 1) * alpha + 1);

        for (size_t i = 0; i < grid.size() - 1; ++i) {
            double start = grid[i];
            double step = (grid[i + 1] - grid[i]) / alpha;
            for (int j = 0; j < alpha; ++j) {
                refined[i * alpha + j] = start + j * step;
            }
        }
        refined[refined.size() - 1] = grid[grid.size() - 1];
        return refined;
    }

    double bisect(double a, double b, double fa, double eps_x) const {
        while (true) {
            double mid = static_cast<double>(midpoint(a, b));
            if (closeenough(a, mid, eps_x)) {
                return mid;
            }

            double fmid = (*this)(mid);
            if (std::signbit(fa) != std::signbit(fmid)) {
                b = mid;
            } else {
                a = mid;
                fa = fmid;
            }
        }
    }

    // Roots function
    Eigen::VectorXd roots(double tol = 1e-10) const
    {
        Eigen::VectorXd grid = this->knots;

        Eigen::VectorXd refined_grid = refine_grid(grid, 2);
        auto f = [this](double x) { return this->operator()(x); };
        // convert to std::vector<double>
        std::vector<double> refined_grid_vec(refined_grid.data(),
                                             refined_grid.data() +
                                                 refined_grid.size());
        std::vector<double> roots = find_all(f, refined_grid_vec);
        return Eigen::Map<Eigen::VectorXd>(roots.data(), roots.size());
    }

    // Overloaded operators
    PiecewiseLegendrePoly operator*(double factor) const
    {
        Eigen::MatrixXd new_data = data * factor;
        return PiecewiseLegendrePoly(new_data, *this, symm);
    }

    friend PiecewiseLegendrePoly operator*(double factor,
                                           const PiecewiseLegendrePoly &poly)
    {
        return poly * factor;
    }

    PiecewiseLegendrePoly operator+(const PiecewiseLegendrePoly &other) const
    {
        if (!knots.isApprox(other.knots, 1e-12)) {
            throw std::runtime_error("knots must be the same");
        }
        Eigen::MatrixXd new_data = data + other.data;
        int new_symm = (symm == other.symm) ? symm : 0;
        return PiecewiseLegendrePoly(new_data, knots, -1, delta_x, new_symm);
    }

    PiecewiseLegendrePoly operator-() const
    {
        Eigen::MatrixXd new_data = -data;
        return PiecewiseLegendrePoly(new_data, knots, -1, delta_x, symm);
    }

    PiecewiseLegendrePoly operator-(const PiecewiseLegendrePoly &other) const
    {
        return (*this) + (-other);
    }

    // Accessor functions
    double get_xmin() const { return xmin; }
    double get_xmax() const { return xmax; }
    const Eigen::VectorXd &get_knots() const { return knots; }
    const Eigen::VectorXd &get_delta_x() const { return delta_x; }
    int get_symm() const { return symm; }
    const Eigen::MatrixXd &get_data() const { return data; }
    const Eigen::VectorXd &get_norms() const { return norms; }
    int get_polyorder() const { return polyorder; }

    /*
    // Helper function to compute legval
    static double legval(double x, const Eigen::VectorXd &coeffs)
    {
        int N = coeffs.size();
        if (N == 0)
            return 0.0;
        std::vector<double> P(N);
        P[0] = 1.0;
        if (N > 1)
            P[1] = x;
        for (int n = 2; n < N; ++n) {
            P[n] = ((2 * n - 1) * x * P[n - 1] - (n - 1) * P[n - 2]) / n;
        }
        double result = 0.0;
        for (int n = 0; n < N; ++n) {
            result += coeffs[n] * P[n];
        }
        return result;
    }
    */

    // Helper function to split x into segment index i and x_tilde
    std::pair<int, double> split(double x) const
    {
        if (x < xmin || x > xmax) {
            throw std::domain_error("x is outside the domain");
        }

        auto it =
            std::lower_bound(knots.data(), knots.data() + knots.size(), x);
        int i = std::max(0, int(it - knots.data() - 1));
        i = std::min<int>(i, knots.size() - 2);

        double x_tilde = (x - xm[i]) * inv_xs[i];
        return std::make_pair(i, x_tilde);
    }

private:
    // Placeholder for Gauss-Legendre quadrature over [a, b]
    double gauss_legendre_quadrature(double a, double b,
                                     std::function<double(double)> f) const
    {
        static const double xg[] = {-0.906179845938664, -0.538469310105683, 0.0,
                                    0.538469310105683, 0.906179845938664};
        static const double wg[] = {0.236926885056189, 0.478628670499366,
                                    0.568888888888889, 0.478628670499366,
                                    0.236926885056189};
        double c1 = (b - a) / 2.0;
        double c2 = (b + a) / 2.0;
        double integral = 0.0;
        for (int j = 0; j < 5; ++j) {
            double x = c1 * xg[j] + c2;
            integral += wg[j] * f(x);
        }
        integral *= c1;
        return integral;
    }

    // Placeholder for finding roots in an interval
    std::vector<double> find_roots_in_interval(std::function<double(double)> f,
                                               double a, double b,
                                               double tol) const
    {
        // Implement a root-finding algorithm like bisection or Brent's method
        std::vector<double> roots;
        // Placeholder: actual implementation needed
        return roots;
    }
};

/*
Eigen::VectorXd derivs(PiecewiseLegendrePoly ppoly, double x){
    std::vector<double> res;
    res.push_back(ppoly(x));

    PiecewiseLegendrePoly ppnew = ppoly;
    for (int i = 2; i <= ppoly.polyorder; i++){
        ppnew = ppnew.deriv();
        res.push_back(ppnew(x));
    }
    return Eigen::Map<Eigen::VectorXd>(res.data(), res.size());
}
*/

} // namespace sparseir

namespace sparseir {

inline std::complex<double> get_tnl(int l, double w)
{
    double abs_w = std::abs(w);
    // Compute spherical Bessel function of order l at abs_w
    double sph_bessel = sphericalbesselj(l, abs_w);
    // Compute 2i^l
    std::complex<double> im_unit(0.0, 1.0);
    std::complex<double> im_power = std::pow(im_unit, l);
    std::complex<double> result = 2.0 * im_power * sph_bessel;

    if (w < 0.0) {
        return std::conj(result);
    } else {
        return result;
    }
}

inline std::pair<std::vector<double>, std::vector<int>>
shift_xmid(const std::vector<double> &knots, const std::vector<double> &delta_x)
{
    size_t N = delta_x.size();
    std::vector<double> delta_x_half(N);
    for (size_t i = 0; i < N; ++i) {
        delta_x_half[i] = delta_x[i] / 2.0;
    }

    // Compute xmid_m1
    std::vector<double> xmid_m1(N);
    std::vector<double> cumsum(N);
    cumsum[0] = delta_x[0];
    for (size_t i = 1; i < N; ++i) {
        cumsum[i] = cumsum[i - 1] + delta_x[i];
    }
    for (size_t i = 0; i < N; ++i) {
        xmid_m1[i] = cumsum[i] - delta_x_half[i];
    }

    // Compute xmid_p1
    std::vector<double> xmid_p1(N);
    std::vector<double> rev_delta_x(N);
    for (size_t i = 0; i < N; ++i) {
        rev_delta_x[N - 1 - i] = delta_x[i];
    }
    std::vector<double> rev_cumsum(N);
    rev_cumsum[0] = rev_delta_x[0];
    for (size_t i = 1; i < N; ++i) {
        rev_cumsum[i] = rev_cumsum[i - 1] + rev_delta_x[i];
    }
    for (size_t i = 0; i < N; ++i) {
        xmid_p1[i] = -rev_cumsum[N - 1 - i] + delta_x_half[i];
    }

    // Compute xmid_0
    std::vector<double> xmid_0(N);
    for (size_t i = 0; i < N; ++i) {
        xmid_0[i] =
            knots[i + 1] - delta_x_half[i]; // Assuming knots has length N + 1
    }

    // Compute shift
    std::vector<int> shift(N);
    for (size_t i = 0; i < N; ++i) {
        shift[i] = static_cast<int>(std::round(xmid_0[i]));
    }

    // Compute diff
    std::vector<double> diff(N);
    for (size_t i = 0; i < N; ++i) {
        int idx = shift[i] + 1; // shift can be -1, 0, 1; idx ranges from 0 to 2
        if (idx == 0) {
            diff[i] = xmid_m1[i];
        } else if (idx == 1) {
            diff[i] = xmid_0[i];
        } else if (idx == 2) {
            diff[i] = xmid_p1[i];
        } else {
            // Should not happen
            throw std::runtime_error("Invalid shift value");
        }
    }

    return std::make_pair(diff, shift);
}

inline Eigen::VectorXcd phase_stable(const PiecewiseLegendrePoly &poly, int wn)
{
    const std::vector<double> knots(poly.knots.data(),
                                    poly.knots.data() + poly.knots.size());
    const std::vector<double> delta_x(
        poly.delta_x.data(), poly.delta_x.data() + poly.delta_x.size());

    // Compute xmid_diff and extra_shift
    std::pair<std::vector<double>, std::vector<int>> shift_result =
        shift_xmid(knots, delta_x);
    const std::vector<double> &xmid_diff = shift_result.first;
    const std::vector<int> &extra_shift = shift_result.second;

    size_t N = delta_x.size();
    Eigen::VectorXcd phase_wi(N);

    for (size_t i = 0; i < N; ++i) {
        int exponent = wn * (extra_shift[i] + 1);
        int exponent_mod4 = ((exponent % 4) + 4) % 4; // Ensure positive modulo

        std::complex<double> im_power;
        switch (exponent_mod4) {
        case 0:
            im_power = std::complex<double>(1.0, 0.0);
            break;
        case 1:
            im_power = std::complex<double>(0.0, 1.0);
            break;
        case 2:
            im_power = std::complex<double>(-1.0, 0.0);
            break;
        case 3:
            im_power = std::complex<double>(0.0, -1.0);
            break;
        }

        double arg = M_PI * wn * xmid_diff[i] / 2.0;
        std::complex<double> cispi = std::polar(1.0, arg); // exp(i * arg)

        phase_wi(i) = im_power * cispi;
    }

    return phase_wi;
}

inline std::complex<double> compute_unl_inner(const PiecewiseLegendrePoly &poly,
                                              int wn)
{
    double wred = M_PI / 4.0 * wn;
    Eigen::VectorXcd phase_wi = phase_stable(poly, wn);

    std::complex<double> res(0.0, 0.0);

    int num_orders = poly.get_data().rows();
    int num_j = poly.get_data().cols();

    for (int order = 0; order < num_orders; ++order) {
        int l = order;
        for (int j = 0; j < num_j; ++j) {
            double data_value = poly.get_data()(order, j);
            double delta_x_j = poly.delta_x(j);
            double norm_j = poly.norms(j);

            double wred_delta_x = wred * delta_x_j;
            std::complex<double> tnl = get_tnl(l, wred_delta_x);
            std::complex<double> phase = phase_wi(j);

            res += data_value * tnl * phase / norm_j;
        }
    }

    res /= std::sqrt(2.0);

    return res;
}
} // namespace sparseir

namespace sparseir {

class PiecewiseLegendrePolyVector {
public:
    // Member variable
    std::vector<PiecewiseLegendrePoly> polyvec;

public:
    // Default constructor
    PiecewiseLegendrePolyVector() = default;

    // Constructor with a vector of PiecewiseLegendrePoly
    explicit PiecewiseLegendrePolyVector(
        const std::vector<PiecewiseLegendrePoly> &polyvec)
        : polyvec(polyvec)
    {
    }

    // Constructor with a 3D array, knots, and symmetry vector
    PiecewiseLegendrePolyVector(const Eigen::Tensor<double, 3> &data3d,
                                const Eigen::VectorXd &knots,
                                const std::vector<int> &symm = {})
        : polyvec(data3d.size())
    {
        if (!symm.empty() && symm.size() != data3d.size()) {
            throw std::invalid_argument("Sizes of data and symm don't match");
        }
        for (auto i = 0; i < data3d.size(); ++i) {
            Eigen::MatrixXd data(data3d.dimension(0), data3d.dimension(1));
            for (auto j = 0; j < data3d.dimension(0); ++j) {
                for (auto k = 0; k < data3d.dimension(1); ++k) {
                    data(j, k) = data3d(j, k, i);
                }
            }

            Eigen::VectorXd delta_x = (knots.tail(knots.size() - 1) - knots.head(knots.size() - 1)).matrix();
            polyvec[i] =
                PiecewiseLegendrePoly(data, knots, static_cast<int>(i),
                                     delta_x, symm.empty() ? 0 : symm[i]);
        }
    }

    /*
    function PiecewiseLegendrePolyVector(polys::PiecewiseLegendrePolyVector,
            knots::AbstractVector; Δx=diff(knots), symm=0)
        length(polys) == length(symm) ||
            throw(DimensionMismatch("Sizes of polys and symm don't match"))

        PiecewiseLegendrePolyVector(map(zip(polys, symm)) do (poly, sym)
            PiecewiseLegendrePoly(poly.data, knots, poly.l; Δx, symm=sym)
        end)
    end
    */
    PiecewiseLegendrePolyVector(const PiecewiseLegendrePolyVector &polys,
                                const Eigen::VectorXd &knots,
                                const Eigen::VectorXd &Δx,
                                const Eigen::VectorXi &symm = Eigen::VectorXi())
        : polyvec(polys.size())
    {
        if (polys.size() != symm.size()) {
            throw std::invalid_argument(
                "Sizes of polys and symm don't match " + std::to_string(polys.size()) + " " + std::to_string(symm.size()));
        }
        for (size_t i = 0; i < polys.size(); ++i) {
            polyvec[i] = PiecewiseLegendrePoly(polys[i].get_data(), knots,
                                               polys[i].l, Δx, symm(i));
        }
    }

    /*
    function PiecewiseLegendrePolyVector(data::AbstractArray{T,3},
        polys::PiecewiseLegendrePolyVector) where {T}
    size(data, 3) == length(polys) ||
        throw(DimensionMismatch("Sizes of data and polys don't match"))

    PiecewiseLegendrePolyVector(map(eachindex(polys)) do i
        PiecewiseLegendrePoly(data[:, :, i], polys[i])
    end)
    }
    */
    PiecewiseLegendrePolyVector(const Eigen::Tensor<double, 3> &data,
                                const PiecewiseLegendrePolyVector &polys)
    {
        std::vector<PiecewiseLegendrePoly> polyvec;
        if (data.dimension(2) != polys.size()) {
            throw std::invalid_argument("Sizes of data and polys don't match");
        }
        for (size_t i = 0; i < data.dimension(2); ++i) {
            Eigen::MatrixXd data2d(data.dimension(0), data.dimension(1));
            for (int j = 0; j < data.dimension(0); ++j) {
                for (int k = 0; k < data.dimension(1); ++k) {
                    data2d(j, k) = data(j, k, i);
                }
            }
            auto p = PiecewiseLegendrePoly(data2d, polys[i]);
            polyvec.push_back(p);
        }
        this->polyvec = polyvec;
    }


    // Add iterator support
    using iterator = std::vector<PiecewiseLegendrePoly>::iterator;
    using const_iterator = std::vector<PiecewiseLegendrePoly>::const_iterator;

    // Iterator methods
    iterator begin() { return polyvec.begin(); }
    iterator end() { return polyvec.end(); }
    const_iterator begin() const { return polyvec.begin(); }
    const_iterator end() const { return polyvec.end(); }

    // Accessors
    size_t size() const { return polyvec.size(); }

    const PiecewiseLegendrePoly &operator[](size_t i) const
    {
        return polyvec[i];
    }

    PiecewiseLegendrePoly &operator[](size_t i) { return polyvec[i]; }

    // Functions to mimic Julia's property accessors
    double xmin() const { return polyvec.empty() ? 0.0 : polyvec[0].xmin; }
    double xmax() const { return polyvec.empty() ? 0.0 : polyvec[0].xmax; }
    Eigen::VectorXd get_knots() const
    {
        return polyvec.empty() ? Eigen::VectorXd() : polyvec[0].knots;
    }
    Eigen::VectorXd get_delta_x() const
    {
        return polyvec.empty() ? Eigen::VectorXd() : polyvec[0].delta_x;
    }
    int get_polyorder() const
    {
        return polyvec.empty() ? 0 : polyvec[0].polyorder;
    }
    Eigen::VectorXd get_norms() const
    {
        return polyvec.empty() ? Eigen::VectorXd() : polyvec[0].norms;
    }

    std::vector<int> get_symm() const
    {
        std::vector<int> symms(polyvec.size());
        for (size_t i = 0; i < polyvec.size(); ++i) {
            symms[i] = polyvec[i].symm;
        }
        return symms;
    }

    // Function to retrieve data as Eigen Tensor
    Eigen::Tensor<double, 3> get_data() const
    {
        if (polyvec.empty()) {
            return Eigen::Tensor<double, 3>();
        }
        int nrows = polyvec[0].data.rows();
        int ncols = polyvec[0].data.cols();
        int npolys = polyvec.size();
        Eigen::Tensor<double, 3> data(nrows, ncols, npolys);
        for (int i = 0; i < npolys; ++i) {
            for (int r = 0; r < nrows; ++r) {
                for (int c = 0; c < ncols; ++c) {
                    data(r, c, i) = polyvec[i].data(r, c);
                }
            }
        }
        return data;
    }

    // Evaluate the vector of polynomials at x
    Eigen::VectorXd operator()(double x) const
    {
        Eigen::VectorXd results(polyvec.size());
        for (size_t i = 0; i < polyvec.size(); ++i) {
            results[i] = polyvec[i](x);
        }
        return results;
    }

    // Evaluate the vector of polynomials at multiple x
    Eigen::MatrixXd operator()(const Eigen::VectorXd &xs) const
    {
        Eigen::MatrixXd results(polyvec.size(), xs.size());
        for (size_t i = 0; i < polyvec.size(); ++i) {
            results.row(i) = polyvec[i](xs);
        }
        return results;
    }
    /*

    // Overlap function
    std::vector<double> overlap(std::function<double(double)> f, double rtol =
    std::numeric_limits<double>::epsilon(), bool return_error = false) const {
        std::vector<double> results;
        for (const auto& poly : polyvec) {
            double integral = poly.overlap(f, rtol, return_error);
            results.push_back(integral);
        }
        return results;
    }

    // Output function
    friend std::ostream& operator<<(std::ostream& os, const
    PiecewiseLegendrePolyVector& polys) { os << polys.size() << "-element
    PiecewiseLegendrePolyVector "; os << "on [" << polys.xmin() << ", " <<
    polys.xmax() << "]"; return os;
    }
    */
};
} // namespace sparseir

namespace sparseir {

// Forward declarations
class PiecewiseLegendrePoly;
class Statistics;

// PowerModel class template
template <typename T>
class PowerModel {
public:
    Eigen::VectorXd moments;
    // Default constructor
    PowerModel() = default;
    // Constructor with moments
    PowerModel(const Eigen::VectorXd &moments_) : moments(moments_) { }
};

/*
function power_moments!(stat, deriv_x1, l)
    statsign = zeta(stat) == 1 ? -1 : 1
    @inbounds for m in 1:length(deriv_x1)
        deriv_x1[m] *= -(statsign * (-1)^m + (-1)^l) / sqrt(2)
    end
    deriv_x1
end
*/
inline Eigen::VectorXd power_moments(const Statistics &stat,
                                      Eigen::VectorXd &deriv_x1, int l)
{
    int statsign = stat.zeta() == 1 ? -1 : 1;
    for (int m = 0; m < deriv_x1.size(); m++) {
        deriv_x1[m] = deriv_x1[m] *
            -(statsign * std::pow(-1, m+1) + std::pow(-1, l)) / std::sqrt(2);
    }
    return deriv_x1;
}

/*
inline PowerModel<double> power_model(const Statistics &stat,
                               const PiecewiseLegendrePoly &poly)
{
    Eigen::VectorXd deriv_x1 = poly.derivs(1.0);
    Eigen::VectorXd moments = power_moments_inplace(stat, deriv_x1, poly.l);
    return PowerModel<double>(moments);
}
*/

// Bosonic and Fermionic statistics classes
class BosonicStatistics : public Statistics {
public:
    int zeta() const override { return 1; }
    bool allowed(int n) const override { return n % 2 == 0 && n != 0; }
};

// PiecewiseLegendreFT class template
template <typename S>
class PiecewiseLegendreFT {
public:
    PiecewiseLegendrePoly poly;
    double n_asymp;
    PowerModel<double> model;
    PiecewiseLegendreFT(const PiecewiseLegendrePoly &poly_,
                        const S &stat,
                        double n_asymp_ = std::numeric_limits<double>::infinity())
        : poly(poly_), n_asymp(n_asymp_)
    {
        if (poly.xmin != -1.0 || poly.xmax != 1.0) {
            throw std::invalid_argument("Only interval [-1, 1] is supported");
        }
        this->model = power_model(stat, poly);
    }

    double get_n_asymp() const { return n_asymp; }
    int zeta() const
    {
        return static_cast<const Statistics &>(S()).zeta();
    }
    const PiecewiseLegendrePoly &get_poly() const { return poly; }

    // Overload operator() for MatsubaraFreq
    std::complex<double>
    operator()(const MatsubaraFreq<S> &omega) const
    {
        int n = static_cast<int>(omega.n);
        if (std::abs(n) < n_asymp) {
            return compute_unl_inner(poly, n);
        } else {
            return giw(*this, n);
        }
    }

    // Overload operator() for integer frequency
    std::complex<double> operator()(int n) const
    {
        return (*this)(MatsubaraFreq<S>(n));
    }

    // Overload operator() for a vector of frequencies
    template <typename Container>
    std::vector<std::complex<double>> operator()(const Container &ns) const
    {
        std::vector<std::complex<double>> res;
        res.reserve(ns.size());
        for (const auto &n : ns) {
            res.push_back((*this)(n));
        }
        return res;
    }

    inline PowerModel<double> power_model(const S &stat, const PiecewiseLegendrePoly &poly)
    {
        Eigen::VectorXd deriv_x1 = poly.derivs(1.0);
        Eigen::VectorXd moments = power_moments(stat, deriv_x1, poly.l);
        return PowerModel<double>(moments);
    }


private:
    // Function to compute the Fourier transform for low frequencies
    std::complex<double> compute_unl_inner(const PiecewiseLegendrePoly &poly,
                                           int wn) const;

    // Function to compute the asymptotic model for high frequencies
    std::complex<double> giw(const PiecewiseLegendreFT &polyFT, int wn) const;

    // Function to evaluate a polynomial at a complex point
    std::complex<double> evalpoly(const std::complex<double> &x,
                                  const std::vector<double> &coeffs) const;
};

// Implementations of member functions

template <typename StatisticsType>
std::complex<double> PiecewiseLegendreFT<StatisticsType>::compute_unl_inner(
    const PiecewiseLegendrePoly &poly, int wn) const
{
    double wred = M_PI / 4.0 * wn;
    Eigen::VectorXcd phase_wi = phase_stable(poly, wn);
    std::complex<double> res = 0.0;

    int order_max = poly.data.rows();
    int segment_count = poly.data.cols();
    for (int order = 0; order < order_max; ++order) {
        for (int j = 0; j < segment_count; ++j) {
            double data_oj = poly.data(order, j);
            std::complex<double> tnl = get_tnl(order, wred * poly.delta_x(j));
            res += data_oj * tnl * phase_wi(j) / poly.norms(j);
        }
    }
    return res / std::sqrt(2.0);
}

template <typename StatisticsType>
std::complex<double>
PiecewiseLegendreFT<StatisticsType>::giw(const PiecewiseLegendreFT &polyFT,
                                        int wn) const
{
    std::complex<double> iw(0.0, M_PI / 2.0 * wn);
    if (wn == 0)
        return std::complex<double>(0.0, 0.0);
    std::complex<double> inv_iw = 1.0 / iw;
    // Convert Eigen::VectorXd to std::vector<double>
    std::vector<double> moments_vec(model.moments.data(),
                                  model.moments.data() + model.moments.size());
    std::complex<double> result = inv_iw * evalpoly(inv_iw, moments_vec);
    return result;
}

template <typename StatisticsType>
inline std::complex<double> PiecewiseLegendreFT<StatisticsType>::evalpoly(
    const std::complex<double> &x, const std::vector<double> &coeffs) const
{
    std::complex<double> result(0, 0);
    for (int i = coeffs.size() - 1; i >= 0; --i) {
        result = result * x + coeffs[i];
    }
    return result;
}

class FermionicStatistics : public Statistics {
public:
    int zeta() const override { return -1; }
    bool allowed(int n) const override { return n % 2 != 0; }
};

template <typename S>
std::function<double(int)> func_for_part(const PiecewiseLegendreFT<S> &polyFT, std::function<double(std::complex<double>)> part = nullptr)
{
    if (part == nullptr) {
        int parity = polyFT.poly.get_symm();
        if (parity == 1) {
            part = std::is_same<S, Bosonic>::value ?
                    [](std::complex<double> x) { return x.real(); } :
                    [](std::complex<double> x) { return x.imag(); };
        }
        else if (parity == -1) {
            part = std::is_same<S, Bosonic>::value ?
                    [](std::complex<double> x) { return x.imag(); } :
                    [](std::complex<double> x) { return x.real(); };
        }
        else {
            throw std::runtime_error("Cannot detect parity");
        }
    }

    return [polyFT, part](int n) -> double {
        auto omega = MatsubaraFreq<S>(2 * n + polyFT.zeta());
        return part(polyFT(omega));
    };
}

// In Julia:
// julia> const DEFAULT_GRID = [range(0; length=2^6); trunc.(Int, exp2.(range(6, 25; length=32 * (25 - 6) + 1)))]
// In C++:
const std::vector<int> DEFAULT_GRID = {
    0,        1,        2,        3,        4,        5,        6,
    7,        8,        9,        10,       11,       12,       13,
    14,       15,       16,       17,       18,       19,       20,
    21,       22,       23,       24,       25,       26,       27,
    28,       29,       30,       31,       32,       33,       34,
    35,       36,       37,       38,       39,       40,       41,
    42,       43,       44,       45,       46,       47,       48,
    49,       50,       51,       52,       53,       54,       55,
    56,       57,       58,       59,       60,       61,       62,
    63,       64,       65,       66,       68,       69,       71,
    72,       74,       76,       77,       79,       81,       82,
    84,       86,       88,       90,       92,       94,       96,
    98,       100,      103,      105,      107,      109,      112,
    114,      117,      119,      122,      125,      128,      130,
    133,      136,      139,      142,      145,      148,      152,
    155,      158,      162,      165,      169,      173,      177,
    181,      184,      189,      193,      197,      201,      206,
    210,      215,      219,      224,      229,      234,      239,
    245,      250,      256,      261,      267,      273,      279,
    285,      291,      297,      304,      311,      317,      324,
    331,      339,      346,      354,      362,      369,      378,
    386,      394,      403,      412,      421,      430,      439,
    449,      459,      469,      479,      490,      501,      512,
    523,      534,      546,      558,      570,      583,      595,
    608,      622,      635,      649,      663,      678,      693,
    708,      724,      739,      756,      772,      789,      806,
    824,      842,      861,      879,      899,      918,      939,
    959,      980,      1002,     1024,     1046,     1069,     1092,
    1116,     1141,     1166,     1191,     1217,     1244,     1271,
    1299,     1327,     1357,     1386,     1417,     1448,     1479,
    1512,     1545,     1579,     1613,     1649,     1685,     1722,
    1759,     1798,     1837,     1878,     1919,     1961,     2004,
    2048,     2092,     2138,     2185,     2233,     2282,     2332,
    2383,     2435,     2488,     2543,     2599,     2655,     2714,
    2773,     2834,     2896,     2959,     3024,     3090,     3158,
    3227,     3298,     3370,     3444,     3519,     3596,     3675,
    3756,     3838,     3922,     4008,     4096,     4185,     4277,
    4371,     4466,     4564,     4664,     4766,     4870,     4977,
    5086,     5198,     5311,     5428,     5547,     5668,     5792,
    5919,     6049,     6181,     6316,     6455,     6596,     6741,
    6888,     7039,     7193,     7351,     7512,     7676,     7844,
    8016,     8192,     8371,     8554,     8742,     8933,     9129,
    9328,     9533,     9741,     9955,     10173,    10396,    10623,
    10856,    11094,    11336,    11585,    11838,    12098,    12363,
    12633,    12910,    13193,    13482,    13777,    14078,    14387,
    14702,    15024,    15353,    15689,    16032,    16384,    16742,
    17109,    17484,    17866,    18258,    18657,    19066,    19483,
    19910,    20346,    20792,    21247,    21712,    22188,    22673,
    23170,    23677,    24196,    24726,    25267,    25820,    26386,
    26964,    27554,    28157,    28774,    29404,    30048,    30706,
    31378,    32065,    32768,    33485,    34218,    34968,    35733,
    36516,    37315,    38132,    38967,    39821,    40693,    41584,
    42494,    43425,    44376,    45347,    46340,    47355,    48392,
    49452,    50535,    51641,    52772,    53928,    55108,    56315,
    57548,    58809,    60096,    61412,    62757,    64131,    65536,
    66971,    68437,    69936,    71467,    73032,    74631,    76265,
    77935,    79642,    81386,    83168,    84989,    86850,    88752,
    90695,    92681,    94711,    96785,    98904,    101070,   103283,
    105545,   107856,   110217,   112631,   115097,   117618,   120193,
    122825,   125514,   128263,   131072,   133942,   136875,   139872,
    142935,   146064,   149263,   152531,   155871,   159284,   162772,
    166337,   169979,   173701,   177504,   181391,   185363,   189422,
    193570,   197809,   202140,   206566,   211090,   215712,   220435,
    225262,   230195,   235236,   240387,   245650,   251029,   256526,
    262144,   267884,   273750,   279744,   285870,   292129,   298526,
    305063,   311743,   318569,   325545,   332674,   339958,   347402,
    355009,   362783,   370727,   378845,   387141,   395618,   404281,
    413133,   422180,   431424,   440871,   450525,   460390,   470472,
    480774,   491301,   502059,   513053,   524288,   535768,   547500,
    559488,   571740,   584259,   597053,   610126,   623487,   637139,
    651091,   665348,   679917,   694805,   710019,   725567,   741455,
    757690,   774282,   791236,   808562,   826267,   844360,   862849,
    881743,   901051,   920781,   940944,   961548,   982603,   1004119,
    1026107,  1048576,  1071536,  1095000,  1118977,  1143480,  1168519,
    1194106,  1220253,  1246974,  1274279,  1302182,  1330696,  1359834,
    1389611,  1420039,  1451134,  1482910,  1515381,  1548564,  1582473,
    1617125,  1652535,  1688721,  1725699,  1763487,  1802102,  1841563,
    1881888,  1923096,  1965207,  2008239,  2052214,  2097152,  2143073,
    2190000,  2237955,  2286960,  2337038,  2388212,  2440507,  2493948,
    2548558,  2604364,  2661392,  2719669,  2779222,  2840079,  2902269,
    2965820,  3030763,  3097128,  3164947,  3234250,  3305071,  3377443,
    3451399,  3526975,  3604205,  3683127,  3763777,  3846193,  3930414,
    4016479,  4104428,  4194304,  4286147,  4380001,  4475911,  4573920,
    4674076,  4776425,  4881015,  4987896,  5097116,  5208729,  5322785,
    5439339,  5558445,  5680159,  5804538,  5931641,  6061527,  6194257,
    6329894,  6468501,  6610142,  6754886,  6902798,  7053950,  7208411,
    7366255,  7527555,  7692387,  7860828,  8032958,  8208857,  8388608,
    8572294,  8760003,  8951822,  9147841,  9348153,  9552851,  9762031,
    9975792,  10194233, 10417458, 10645571, 10878678, 11116890, 11360318,
    11609077, 11863283, 12123055, 12388515, 12659788, 12937002, 13220285,
    13509772, 13805597, 14107900, 14416823, 14732510, 15055110, 15384774,
    15721657, 16065917, 16417714, 16777216, 17144589, 17520006, 17903645,
    18295683, 18696307, 19105702, 19524063, 19951584, 20388467, 20834916,
    21291142, 21757357, 22233781, 22720637, 23218155, 23726566, 24246110,
    24777031, 25319577, 25874004, 26440571, 27019544, 27611195, 28215801,
    28833647, 29465021, 30110221, 30769549, 31443315, 32131834, 32835429,
    33554432,
};

// If the set of Matsubara points is not symmetric, modify the vector by
// removing zero (if present) and prepending the negatives of the reversed list.
inline void symmetrize_matsubara_inplace(std::vector<int> &xs)
{
    // is sorted
    if (!std::is_sorted(xs.begin(), xs.end())) {
        throw std::runtime_error("points must be sorted");
    }
    if (xs.empty())
        return;
    if (xs.front() < 0)
        throw std::runtime_error("points must be non-negative");

    std::vector<int> neg(xs.rbegin(), xs.rend());
    for (auto &x : neg){
        x = -x;
    }
    if (std::abs(xs.front()) < 1e-12 && !xs.empty()){
        xs.erase(xs.begin());
    }
    xs.insert(xs.begin(), neg.begin(), neg.end());
}

template <typename S>
std::vector<MatsubaraFreq<S>> sign_changes(const PiecewiseLegendreFT<S> &u_hat, bool positive_only=false)
{
    auto grid = DEFAULT_GRID;
    auto f = func_for_part(u_hat);
    auto x0 = find_all(f, grid);
    for (std::size_t i = 0; i < x0.size(); i++) {
        x0[i] = 2 * x0[i] + u_hat.zeta();
    }

    if (!positive_only) {
        symmetrize_matsubara_inplace(x0);
    }

    std::vector<MatsubaraFreq<S>> result;
    for (auto x : x0) {
        result.push_back(MatsubaraFreq<S>(x));
    }
    return result;
}

template <typename S>
std::vector<MatsubaraFreq<S>> find_extrema(const PiecewiseLegendreFT<S> &u_hat, bool positive_only=false)
{
    auto f = func_for_part(u_hat);
    auto x0 = discrete_extrema(f, DEFAULT_GRID);
    for (auto &x : x0) {
        x = 2 * x + u_hat.zeta();
    }
    if (!positive_only) {
        symmetrize_matsubara_inplace(x0);
    }
    std::vector<MatsubaraFreq<S>> results;
    for (auto x : x0) {
        results.push_back(MatsubaraFreq<S>(x));
    }
    return results;
}

// Evaluate a polynomial (e.g. in inv_iw) with given coefficients (using
// Horner's method).
inline std::complex<double> evalpoly(std::complex<double> x,
                                     const std::vector<double> &coeffs)
{
    std::complex<double> result(0, 0);
    std::complex<double> xn(1, 0);
    for (double c : coeffs) {
        result += c * xn;
        xn *= x;
    }
    return result;
}


} // namespace sparseir

namespace sparseir {

// PiecewiseLegendreFTVector class in C++

template <typename S>
class PiecewiseLegendreFTVector {
private:
    std::vector<PiecewiseLegendreFT<S>> polyvec;

public:
    // Default constructor
    PiecewiseLegendreFTVector() = default;

    // Constructor from vector of PiecewiseLegendreFT<S>
    PiecewiseLegendreFTVector<S>(
        const std::vector<PiecewiseLegendreFT<S>> &polyvec_)
        : polyvec(polyvec_)
    {
    }


    // Constructor from PiecewiseLegendrePolyVector and Statistics
    PiecewiseLegendreFTVector<S>(PiecewiseLegendrePolyVector &polys,
                              S &stat,
                              double n_asymp = std::numeric_limits<double>::infinity())
    {
        std::vector<PiecewiseLegendreFT<S>> polyvec_;
        polyvec_.reserve(polys.size());
        for (const auto &poly : polys)
        {
            polyvec_.push_back(PiecewiseLegendreFT<S>(poly, stat, n_asymp));
        }
        polyvec = polyvec_;
    }


    // Get the size of the vector
    size_t size() const { return polyvec.size(); }

    // Indexing operator (non-const)
    PiecewiseLegendreFT<S> &operator[](size_t i) { return polyvec[i]; }

    // Indexing operator (const)
    const PiecewiseLegendreFT<S> &operator[](size_t i) const
    {
        return polyvec[i];
    }

    // Indexing with a vector of indices
    PiecewiseLegendreFTVector<S>
    operator[](const std::vector<size_t> &indices) const
    {
        std::vector<PiecewiseLegendreFT<S>> new_polyvec;
        new_polyvec.reserve(indices.size());
        for (size_t idx : indices) {
            new_polyvec.push_back(polyvec[idx]);
        }
        return PiecewiseLegendreFTVector<S>{std::move(new_polyvec)};
    }

    // Set element at index i
    void set(size_t i, const PiecewiseLegendreFT<S> &p)
    {
        if (i < polyvec.size()) {
            polyvec[i] = p;
        }
    }

    // Create a similar PiecewiseLegendreFTVector
    PiecewiseLegendreFTVector<S> similar() const
    {
        return PiecewiseLegendreFTVector<S>();
    }

    // Get n_asymp from the first element
    double n_asymp() const
    {
        return polyvec.empty() ? std::numeric_limits<double>::infinity()
                               : polyvec.front().n_asymp();
    }

    // Get statistics from the first element
    S statistics() const { return polyvec.front().statistics(); }

    // Get zeta from the first element
    double zeta() const
    {
        return polyvec.empty() ? 0.0 : polyvec.front().zeta();
    }

    /*
    // Get poly as PiecewiseLegendrePolyVector
    PiecewiseLegendrePolyVector<S> poly() const
    {
        std::vector<PiecewiseLegendrePoly<S>> polys;
        polys.reserve(polyvec.size());
        for (const auto &pft : polyvec)
        {
            polys.push_back(pft.poly());
        }
        return PiecewiseLegendrePolyVector<S>{std::move(polys)};
    }
    */

    // Overload operator() for MatsubaraFreq<S>
    Eigen::VectorXcd operator()(const MatsubaraFreq<S> &omega) const
    {
        size_t num_funcs = polyvec.size();
        Eigen::VectorXcd result(num_funcs);
        for (size_t i = 0; i < num_funcs; ++i) {
            result(i) = polyvec[i](omega);
        }
        return result;
    }

    // Overload operator() for integer n
    Eigen::VectorXcd operator()(int n) const
    {
        return (*this)(MatsubaraFreq<S>(n));
    }

    // Overload operator() for array of integers
    Eigen::MatrixXcd operator()(const Eigen::ArrayXi &n_array) const
    {
        size_t num_funcs = polyvec.size();
        size_t num_freqs = n_array.size();
        Eigen::MatrixXcd result(num_funcs, num_freqs);
        for (size_t i = 0; i < num_funcs; ++i) {
            for (size_t j = 0; j < num_freqs; ++j) {
                result(i, j) = polyvec[i](n_array[j]);
            }
        }
        return result;
    }
};
} // namespace sparseir