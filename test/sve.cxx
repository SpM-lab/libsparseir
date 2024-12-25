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
    auto epsilon = std::numeric_limits<double>::quiet_NaN();

    double safe_epsilon;
    std::string Twork_actual;
    std::string svd_strategy_actual;
    std::tie(safe_epsilon, Twork_actual, svd_strategy_actual) = sparseir::auto_choose_accuracy(epsilon, "Float64x2");

    std::cout << Twork_actual << std::endl;
    REQUIRE(Twork_actual == "Float64x2");

    using T = xprec::DDouble;
    auto lk = std::make_shared<sparseir::LogisticKernel<T>>(12.0);

    //auto x = xprec::DDouble(1e-2, 0.0);
    //auto y = xprec::DDouble(1e-2, 0.0);
    //std::cout << "logistic kernel " << (*lk)(x, y) << std::endl;
    //std::cout << "logistic kernel " << (*lk)(x, y, x + 1.0, 1.0 - x) << std::endl;

    auto lkeven = sparseir::ReducedKernel<sparseir::LogisticKernel<T>, T>(lk, 1);

    auto x = xprec::DDouble(5.3168114454740182e-04,4.7935591542849864e-20);
    auto y = xprec::DDouble(7.5138745175870199e-05,4.4864960587403202e-21);
    auto x_forward = xprec::DDouble(5.3168114454740182e-04,4.7935591542850225e-20);
    auto x_backward = xprec::DDouble(9.9946831885545262e-01,-2.1731979041252942e-17);

    std::cout << x + 1 << std::endl;
    std::cout << 1 - x << std::endl;
    x_forward = x + 1;
    auto res = (lkeven)(x, y);
    std::cout << "logistic kernel " << res << std::endl;

    std::cout << std::endl;
    auto res2 =(lkeven)(x, y, x_forward - 1, x_backward);
    std::cout << "logistic kernel " << res2 << std::endl;

    auto sve = sparseir::compute_sve<T>(lk, safe_epsilon);
    //using T = xprec::DDouble;
    //auto lk = std::make_shared<sparseir::LogisticKernel<T>>(12.0);
    //std::cout << "logistic kernel " << (*lk)(0.0, 0.0) << std::endl;
    //std::cout << "logistic kernel " << (*lk)(0.1, 0.1) << std::endl;
    //auto sve = sparseir::compute_sve<sparseir::LogisticKernel<T>>(lk);
    //auto s = sve.s;
    //std::cout << "S values: \n" << s << std::endl;
}