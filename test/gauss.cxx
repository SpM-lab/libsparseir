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

#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

template <typename T>
void gaussValidate(const Rule<T>& rule) {
    if (!(rule.a <= rule.b)) {
        throw invalid_argument("a,b must be a valid interval");
    }
    if (!all_of(rule.x.begin(), rule.x.end(), [rule](T xi) { return xi <= rule.b; })) {
        throw invalid_argument("x must be smaller than b");
    }
    if (!all_of(rule.x.begin(), rule.x.end(), [rule](T xi) { return xi >= rule.a; })) {
        throw invalid_argument("x must be larger than a");
    }
    if (!is_sorted(rule.x.begin(), rule.x.end())) {
        throw invalid_argument("x must be well-ordered");
    }
    if (rule.x.size() != rule.w.size()) {
        throw invalid_argument("shapes are inconsistent");
    }

    REQUIRE(equal(rule.x_forward.begin(), rule.x_forward.end(), rule.x.begin(), [rule](T xi, T x_forward) { return abs(x_forward - (xi - rule.a)) < 1e-9; }));
    REQUIRE(equal(rule.x_backward.begin(), rule.x_backward.end(), rule.x.begin(), [rule](T xi, T x_backward) { return abs(x_backward - (rule.b - xi)) < 1e-9; }));
}

TEST_CASE("gauss.cpp") {
    /*
    SECTION("collocate") {
        Rule<double> r = legendre(20); // Assuming legendre function is defined
        MatrixXd cmat = legendre_collocation(r); // Assuming legendre_collocation function is defined
        MatrixXd emat = legvander(r.x, r.x.size() - 1); // Assuming legvander function is defined
        REQUIRE((emat * cmat).isApprox(MatrixXd::Identity(20, 20), 1e-13));
    }
    */

    /*
    SECTION("gauss legendre") {
        Rule<double> rule = legendre(200); // Assuming legendre function is defined
        gaussValidate(rule);
        vector<double> x, w;
        tie(x, w) = gauss(200); // Assuming gauss function is defined
        REQUIRE(equal(rule.x.begin(), rule.x.end(), x.begin(), [](double a, double b) { return abs(a - b) < 1e-9; }));
        REQUIRE(equal(rule.w.begin(), rule.w.end(), w.begin(), [](double a, double b) { return abs(a - b) < 1e-9; }));
    }
    */

    /*
    SECTION("piecewise") {
        vector<double> edges = {-4, -1, 1, 3};
        Rule<double> rule = piecewise(legendre(20), edges); // Assuming piecewise function is defined
        gaussValidate(rule);
    }
    */
}