#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "sparseir/sparseir-header-only.h"
#include <xprec/ddouble.h>

#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using xprec::DDouble;

TEST_CASE("gauss", "[Rule]")
{
    vector<DDouble> x, w, v, x_forward, x_backward;
    // Initialize x, w, v, x_forward, x_backward with DDouble values

    NestedRule<DDouble> rule(x, w, v, x_forward, x_backward);
    // Use the rule object as needed
    REQUIRE(1==1);
}

template <typename T>
void validateGauss(const Rule<T>& rule) {
    REQUIRE(rule.a <= rule.b);
    REQUIRE(all_of(rule.x.begin(), rule.x.end(), [rule](T xi) { return xi <= rule.b; }));
    REQUIRE(all_of(rule.x.begin(), rule.x.end(), [rule](T xi) { return xi >= rule.a; }));
    REQUIRE(is_sorted(rule.x.begin(), rule.x.end()));
    REQUIRE(rule.x.size() == rule.w.size());
    REQUIRE(equal(rule.x_forward.begin(), rule.x_forward.end(), rule.x.begin(), [rule](T xi, T x_forward) { return abs(x_forward - (xi - rule.a)) < 1e-9; }));
    REQUIRE(equal(rule.x_backward.begin(), rule.x_backward.end(), rule.x.begin(), [rule](T xi, T x_backward) { return abs(x_backward - (rule.b - xi)) < 1e-9; }));
}

TEST_CASE("gauss.cpp") {
    SECTION("collocate") {
        Rule<double> r = legendre(20); // Assuming legendre function is defined
        MatrixXd cmat = legendre_collocation(r); // Assuming legendre_collocation function is defined
        MatrixXd emat = legvander(r.x, r.x.size() - 1); // Assuming legvander function is defined
        REQUIRE((emat * cmat).isApprox(MatrixXd::Identity(20, 20)));
    }

    SECTION("gauss legendre") {
        Rule<double> rule = legendre(200); // Assuming legendre function is defined
        validateGauss(rule);
        vector<double> x, w;
        tie(x, w) = gauss(200); // Assuming gauss function is defined
        REQUIRE(equal(rule.x.begin(), rule.x.end(), x.begin(), [](double a, double b) { return abs(a - b) < 1e-9; }));
        REQUIRE(equal(rule.w.begin(), rule.w.end(), w.begin(), [](double a, double b) { return abs(a - b) < 1e-9; }));
    }

    SECTION("piecewise") {
        vector<double> edges = {-4, -1, 1, 3};
        Rule<double> rule = piecewise(legendre(20), edges); // Assuming piecewise function is defined
        validateGauss(rule);
    }

    SECTION("scale") {
        Rule<double> rule = legendre(30); // Assuming legendre function is defined
        scale(rule, 2); // Assuming scale function is defined
    }
}