// Include necessary headers
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <functional>

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <functional>

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

    // Constructor
    PiecewiseLegendrePoly(int polyorder, double xmin, double xmax,
                          const Eigen::VectorXd& knots,
                          const Eigen::VectorXd& delta_x,
                          const Eigen::MatrixXd& data,
                          int symm, int l,
                          const Eigen::VectorXd& xm,
                          const Eigen::VectorXd& inv_xs,
                          const Eigen::VectorXd& norms)
        : polyorder(polyorder), xmin(xmin), xmax(xmax),
          knots(knots), delta_x(delta_x), data(data), symm(symm), l(l),
          xm(xm), inv_xs(inv_xs), norms(norms)
    {
        // Check for NaN in data
        if (data.unaryExpr([](double x) { return std::isnan(x); }).any()) {
            throw std::runtime_error("data contains NaN");
        }

        // Check that knots are sorted
        if (!std::is_sorted(knots.data(), knots.data() + knots.size())) {
            throw std::runtime_error("knots must be monotonically increasing");
        }

        // Check that delta_x[i] == knots[i+1] - knots[i]
        const double tol = 1e-12;
        for (int i = 0; i < delta_x.size(); ++i) {
            double diff = knots[i + 1] - knots[i];
            if (std::abs(delta_x[i] - diff) > tol) {
                throw std::runtime_error("delta_x must work with knots");
            }
        }
    }

    // Constructor: PiecewiseLegendrePoly(data, p; symm=symm(p))
    PiecewiseLegendrePoly(const Eigen::MatrixXd& data, const PiecewiseLegendrePoly& p, int symm)
        : polyorder(p.polyorder), xmin(p.xmin), xmax(p.xmax),
          knots(p.knots), delta_x(p.delta_x), data(data), symm(symm), l(p.l),
          xm(p.xm), inv_xs(p.inv_xs), norms(p.norms)
    {
        // Copy constructor with new data and symm
    }

    // Constructor: PiecewiseLegendrePoly(data::Matrix, knots::Vector, l::Integer;
    //      delta_x=diff(knots), symm=0)
    PiecewiseLegendrePoly(const Eigen::MatrixXd& data, const Eigen::VectorXd& knots, int l,
                          const Eigen::VectorXd& delta_x = Eigen::VectorXd(),
                          int symm = 0)
        : data(data), knots(knots), symm(symm), l(l)
    {
        polyorder = data.rows();
        int nsegments = data.cols();
        if (knots.size() != nsegments + 1) {
            throw std::runtime_error("Invalid knots array");
        }
        xmin = knots[0];
        xmax = knots[knots.size() - 1];

        if (delta_x.size() == 0) {
            // delta_x = diff(knots)
            this->delta_x = knots.segment(1, knots.size() - 1) - knots.segment(0, knots.size() - 1);
        } else {
            this->delta_x = delta_x;
        }

        // xm = (knots[1:end-1] + knots[2:end]) / 2
        xm = (knots.segment(0, knots.size() - 1) + knots.segment(1, knots.size() - 1)) / 2.0;

        // inv_xs = 2 ./ delta_x
        inv_xs = 2.0 / this->delta_x.array();

        // norms = sqrt.(inv_xs)
        norms = inv_xs.array().sqrt();

        // Check for NaN in data
        if (data.unaryExpr([](double x) { return std::isnan(x); }).any()) {
            throw std::runtime_error("data contains NaN");
        }

        // Check that knots are sorted
        if (!std::is_sorted(knots.data(), knots.data() + knots.size())) {
            throw std::runtime_error("knots must be monotonically increasing");
        }
    }

    // Function call operator: evaluate the polynomial at x
    double operator()(double x) const {
        int i;
        double x_tilde;
        std::tie(i, x_tilde) = split(x);
        Eigen::VectorXd coeffs = data.col(i);
        double value = legval(x_tilde, coeffs) * norms[i];
        return value;
    }

    // Evaluate the polynomial at an array of x
    Eigen::VectorXd operator()(const Eigen::VectorXd& xs) const {
        Eigen::VectorXd results(xs.size());
        for (int idx = 0; idx < xs.size(); ++idx) {
            results[idx] = (*this)(xs[idx]);
        }
        return results;
    }

    // Overlap function
    double overlap(std::function<double(double)> f, double rtol = std::numeric_limits<double>::epsilon(),
                   bool return_error = false, int maxevals = 10000, const std::vector<double>& points = {}) const {
        // Implement numerical integration over the intervals
        // Since C++ does not have a built-in quadgk, we need to implement one or use a library
        // For simplicity, let's use Gauss-Legendre quadrature over each segment
        double result = 0.0;

        std::vector<double> integration_points(knots.data(), knots.data() + knots.size());
        integration_points.insert(integration_points.end(), points.begin(), points.end());
        std::sort(integration_points.begin(), integration_points.end());
        integration_points.erase(std::unique(integration_points.begin(), integration_points.end()), integration_points.end());

        for (size_t idx = 0; idx < integration_points.size() - 1; ++idx) {
            double a = integration_points[idx];
            double b = integration_points[idx + 1];

            // Perform Gauss-Legendre quadrature on [a, b]
            auto integrand = [this, &f](double x) {
                return (*this)(x) * f(x);
            };
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
    PiecewiseLegendrePoly deriv(int n = 1) const {
        Eigen::MatrixXd ddata = legder(data, n);

        // Multiply each column by inv_xs[i]^n
        for (int i = 0; i < ddata.cols(); ++i) {
            ddata.col(i) *= std::pow(inv_xs[i], n);
        }

        int new_symm = std::pow(-1, n) * symm;
        return PiecewiseLegendrePoly(ddata, *this, new_symm);
    }

    // Roots function
    std::vector<double> roots(double tol = 1e-10) const {
        std::vector<double> all_roots;

        // For each segment, find the roots of the polynomial
        for (int i = 0; i < data.cols(); ++i) {
            // Create a function for the polynomial in this segment
            auto segment_poly = [this, i](double x) {
                double x_tilde = (x - xm[i]) * inv_xs[i];
                Eigen::VectorXd coeffs = data.col(i);
                double value = legval(x_tilde, coeffs) * norms[i];
                return value;
            };

            // Find roots in the interval [knots[i], knots[i+1]]
            std::vector<double> segment_roots = find_roots_in_interval(segment_poly, knots[i], knots[i + 1], tol);
            all_roots.insert(all_roots.end(), segment_roots.begin(), segment_roots.end());
        }

        return all_roots;
    }

    // Overloaded operators
    PiecewiseLegendrePoly operator*(double factor) const {
        Eigen::MatrixXd new_data = data * factor;
        return PiecewiseLegendrePoly(new_data, *this, symm);
    }

    friend PiecewiseLegendrePoly operator*(double factor, const PiecewiseLegendrePoly& poly) {
        return poly * factor;
    }

    PiecewiseLegendrePoly operator+ (const PiecewiseLegendrePoly& other) const {
        if (!knots.isApprox(other.knots, 1e-12)) {
            throw std::runtime_error("knots must be the same");
        }
        Eigen::MatrixXd new_data = data + other.data;
        int new_symm = (symm == other.symm) ? symm : 0;
        return PiecewiseLegendrePoly(new_data, knots, -1, delta_x, new_symm);
    }

    PiecewiseLegendrePoly operator- () const {
        Eigen::MatrixXd new_data = -data;
        return PiecewiseLegendrePoly(new_data, knots, -1, delta_x, symm);
    }

    PiecewiseLegendrePoly operator- (const PiecewiseLegendrePoly& other) const {
        return (*this) + (-other);
    }

    // Accessor functions
    double get_xmin() const { return xmin; }
    double get_xmax() const { return xmax; }
    const Eigen::VectorXd& get_knots() const { return knots; }
    const Eigen::VectorXd& get_delta_x() const { return delta_x; }
    int get_symm() const { return symm; }
    const Eigen::MatrixXd& get_data() const { return data; }
    const Eigen::VectorXd& get_norms() const { return norms; }
    int get_polyorder() const { return polyorder; }

private:
    // Helper function to compute legval
    static double legval(double x, const Eigen::VectorXd& coeffs) {
        int N = coeffs.size();
        if (N == 0) return 0.0;
        std::vector<double> P(N);
        P[0] = 1.0;
        if (N > 1) P[1] = x;
        for (int n = 2; n < N; ++n) {
            P[n] = ((2 * n - 1) * x * P[n - 1] - (n - 1) * P[n - 2]) / n;
        }
        double result = 0.0;
        for (int n = 0; n < N; ++n) {
            result += coeffs[n] * P[n];
        }
        return result;
    }

    // Helper function to split x into segment index i and x_tilde
    std::pair<int, double> split(double x) const {
        if (x < xmin || x > xmax) {
            throw std::domain_error("x is outside the domain");
        }

        auto it = std::lower_bound(knots.data(), knots.data() + knots.size(), x);
        int i = std::max(0, int(it - knots.data() - 1));
        i = std::min<int>(i, knots.size() - 2);

        double x_tilde = (x - xm[i]) * inv_xs[i];
        return std::make_pair(i, x_tilde);
    }

    // Placeholder for Gauss-Legendre quadrature over [a, b]
    double gauss_legendre_quadrature(double a, double b, std::function<double(double)> f) const {
        static const double xg[] = { -0.906179845938664, -0.538469310105683,
                                     0.0,
                                     0.538469310105683, 0.906179845938664 };
        static const double wg[] = { 0.236926885056189, 0.478628670499366,
                                     0.568888888888889,
                                     0.478628670499366, 0.236926885056189 };
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
    std::vector<double> find_roots_in_interval(std::function<double(double)> f, double a, double b, double tol) const {
        // Implement a root-finding algorithm like bisection or Brent's method
        std::vector<double> roots;
        // Placeholder: actual implementation needed
        return roots;
    }
};

} // namespace sparseir


/*
namespace sparseir {
    class PiecewiseLegendrePolyVector {
public:
    // Member variable
    std::vector<PiecewiseLegendrePoly> polyvec;

    // Constructors
    PiecewiseLegendrePolyVector() {}

    // Constructor with polyvec
    PiecewiseLegendrePolyVector(const std::vector<PiecewiseLegendrePoly>& polyvec)
        : polyvec(polyvec) {}

    // Constructor with data tensor, knots, and optional symm vector
    PiecewiseLegendrePolyVector(const Eigen::Tensor<double, 3>& data,
                                const Eigen::VectorXd& knots,
                                const std::vector<int>& symm = std::vector<int>())
    {
        int npolys = data.dimension(2);
        if (!symm.empty() && symm.size() != npolys) {
            throw std::runtime_error("Sizes of data and symm don't match");
        }
        polyvec.reserve(npolys);
        for (int i = 0; i < npolys; ++i) {
            Eigen::MatrixXd data_i = data.chip(i, 2);
            int sym = symm.empty() ? 0 : symm[i];
            polyvec.emplace_back(data_i, knots, i, Eigen::VectorXd(), sym);
        }
    }

    // Constructor with polys, new knots, delta_x, and symm
    PiecewiseLegendrePolyVector(const PiecewiseLegendrePolyVector& polys,
                                const Eigen::VectorXd& knots,
                                const Eigen::VectorXd& delta_x = Eigen::VectorXd(),
                                const std::vector<int>& symm = std::vector<int>())
    {
        if (!symm.empty() && symm.size() != polys.size()) {
            throw std::runtime_error("Sizes of polys and symm don't match");
        }
        polyvec.reserve(polys.size());
        for (size_t i = 0; i < polys.size(); ++i) {
            int sym = symm.empty() ? 0 : symm[i];
            polyvec.emplace_back(polys.polyvec[i].data, knots, polys.polyvec[i].l, delta_x, sym);
        }
    }

    // Constructor with data tensor and existing polys
    PiecewiseLegendrePolyVector(const Eigen::Tensor<double, 3>& data,
                                const PiecewiseLegendrePolyVector& polys)
    {
        int npolys = polys.size();
        if (data.dimension(2) != npolys) {
            throw std::runtime_error("Sizes of data and polys don't match");
        }
        polyvec.reserve(npolys);
        for (int i = 0; i < npolys; ++i) {
            Eigen::MatrixXd data_i = data.chip(i, 2);
            polyvec.emplace_back(data_i, polys.polyvec[i]);
        }
    }

    // Accessors
    size_t size() const { return polyvec.size(); }

    const PiecewiseLegendrePoly& operator[](size_t i) const {
        return polyvec[i];
    }

    PiecewiseLegendrePoly& operator[](size_t i) {
        return polyvec[i];
    }

    // Functions to mimic Julia's property accessors
    double xmin() const { return polyvec.empty() ? 0.0 : polyvec[0].xmin; }
    double xmax() const { return polyvec.empty() ? 0.0 : polyvec[0].xmax; }
    Eigen::VectorXd get_knots() const { return polyvec.empty() ? Eigen::VectorXd() : polyvec[0].knots; }
    Eigen::VectorXd get_delta_x() const { return polyvec.empty() ? Eigen::VectorXd() : polyvec[0].delta_x; }
    int get_polyorder() const { return polyvec.empty() ? 0 : polyvec[0].polyorder; }
    Eigen::VectorXd get_norms() const { return polyvec.empty() ? Eigen::VectorXd() : polyvec[0].norms; }

    std::vector<int> get_symm() const {
        std::vector<int> symms(polyvec.size());
        for (size_t i = 0; i < polyvec.size(); ++i) {
            symms[i] = polyvec[i].symm;
        }
        return symms;
    }

    // Function to retrieve data as Eigen Tensor
    Eigen::Tensor<double, 3> get_data() const {
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
    Eigen::VectorXd operator()(double x) const {
        Eigen::VectorXd results(polyvec.size());
        for (size_t i = 0; i < polyvec.size(); ++i) {
            results[i] = polyvec[i](x);
        }
        return results;
    }

    // Evaluate the vector of polynomials at multiple x
    Eigen::MatrixXd operator()(const Eigen::VectorXd& xs) const {
        Eigen::MatrixXd results(polyvec.size(), xs.size());
        for (size_t i = 0; i < polyvec.size(); ++i) {
            results.row(i) = polyvec[i](xs);
        }
        return results;
    }

    // Overlap function
    std::vector<double> overlap(std::function<double(double)> f, double rtol = std::numeric_limits<double>::epsilon(),
                                bool return_error = false) const {
        std::vector<double> results;
        for (const auto& poly : polyvec) {
            double integral = poly.overlap(f, rtol, return_error);
            results.push_back(integral);
        }
        return results;
    }

    // Output function
    friend std::ostream& operator<<(std::ostream& os, const PiecewiseLegendrePolyVector& polys) {
        os << polys.size() << "-element PiecewiseLegendrePolyVector ";
        os << "on [" << polys.xmin() << ", " << polys.xmax() << "]";
        return os;
    }
};
} // namespace sparseir
*/
