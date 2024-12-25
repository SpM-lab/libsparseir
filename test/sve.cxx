// sparseir/tests/sve.cpp

#include <Eigen/Dense>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <map>
#include <numeric>
#include <vector>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

using std::invalid_argument;

// Function to check smoothness
void check_smooth(const std::function<double(double)> &u,
                  const std::vector<double> &s, double uscale,
                  double fudge_factor)
{
    /*
    double epsilon = std::numeric_limits<double>::epsilon();
    std::vector<double> knots = sparseir::knots(u);

    // Remove the first and last knots
    if (knots.size() <= 2) {
        REQUIRE(false); // Not enough knots to perform the test
    }
    knots.erase(knots.begin());
    knots.pop_back();

    std::vector<double> jump(knots.size());
    std::vector<double> compare_below(knots.size());
    std::vector<double> compare_above(knots.size());
    std::vector<double> compare(knots.size());

    for (size_t i = 0; i < knots.size(); ++i) {
        double x = knots[i];
        jump[i] = std::abs(u(x + epsilon) - u(x - epsilon));
        compare_below[i] = std::abs(u(x - epsilon) - u(x - 3 * epsilon));
        compare_above[i] = std::abs(u(x + 3 * epsilon) - u(x + epsilon));
        compare[i] = std::min(compare_below[i], compare_above[i]);
        compare[i] = std::max(compare[i], uscale * epsilon);
        // Loss of precision
        compare[i] *= fudge_factor * (s[0] / s[i]);
    }

    bool all_less = true;
    for (size_t i = 0; i < jump.size(); ++i) {
        if (!(jump[i] < compare[i])) {
            all_less = false;
            break;
        }
    }

    REQUIRE(all_less);
    */
}

/*
TEST_CASE("sve.cpp", "[SamplingSVE]")
{
    //sparseir::LogisticKernel lk(10.0);
    auto lk = std::make_shared<sparseir::LogisticKernel>(10.0);
    REQUIRE(lk->lambda_ == 10.0);
    auto hints = sparseir::sve_hints(lk, 1e-6);
    int nsvals_hint = hints->nsvals();
    int n_gauss = hints->ngauss();
    std::vector<double> segs_x = hints->segments_x();
    std::vector<double> segs_y = hints->segments_y();

    // Ensure `convert` is declared before this line
    sparseir::Rule<double> rule = sparseir::legendre<double>(n_gauss);

    sparseir::Rule<double> gauss_x = rule.piecewise(segs_x);
    sparseir::Rule<double> gauss_y = rule.piecewise(segs_y);
    auto ssve1 = sparseir::SamplingSVE<sparseir::LogisticKernel>(lk, 1e-6);
    REQUIRE(ssve1.n_gauss == n_gauss);
    auto ssve2 = sparseir::SamplingSVE<sparseir::LogisticKernel>(lk, 1e-6, 12);
    REQUIRE(ssve2.n_gauss == 12);

    auto ssve1_double =
        sparseir::SamplingSVE<sparseir::LogisticKernel, double>(lk, 1e-6);
    REQUIRE(ssve1_double.n_gauss == n_gauss);
    auto ssve2_double =
        sparseir::SamplingSVE<sparseir::LogisticKernel, double>(lk, 1e-6, 12);
    REQUIRE(ssve2_double.n_gauss == 12);

    auto ssve1_ddouble =
        sparseir::SamplingSVE<sparseir::LogisticKernel, xprec::DDouble>(lk,
                                                                        1e-6);
    REQUIRE(ssve1_ddouble.n_gauss == n_gauss);
    auto ssve2_ddouble =
        sparseir::SamplingSVE<sparseir::LogisticKernel, xprec::DDouble>(
            lk, 1e-6, 12);
    REQUIRE(ssve2_ddouble.n_gauss == 12);
}
*/


TEST_CASE("compute_sve", "[compute_sve]"){
    //sparseir::LogisticKernel lk(12.0);
    using T = xprec::DDouble;
    auto lk = std::make_shared<sparseir::LogisticKernel<T>>(12.0);
    std::cout << "logistic kernel " << (*lk)(0.0, 0.0) << std::endl;
    std::cout << "logistic kernel " << (*lk)(0.1, 0.1) << std::endl;
    auto sve = sparseir::compute_sve<sparseir::LogisticKernel<T>>(lk);
    auto s = sve.s;
    std::cout << "S values: \n" << s << std::endl;
}