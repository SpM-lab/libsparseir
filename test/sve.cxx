// sparseir/tests/sve.cpp

#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <cstdint>
#include <map>

#include <catch2/catch_test_macros.hpp>

#include <xprec/ddouble-header-only.hpp>
#include <sparseir/sparseir-header-only.hpp>

using std::invalid_argument;

using xprec::DDouble;

// Function to check smoothness
void check_smooth(const std::function<double(double)>& u, const std::vector<double>& s, double uscale, double fudge_factor) {
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

TEST_CASE("sve.cpp", "[sve]") {

    // Define a map to store SVEResult objects
    auto sve_logistic = std::map < int, sparseir::SVEResult<sparseir::LogisticKernel>>{
                                            {10, sparseir::compute_sve<sparseir::LogisticKernel, double>(sparseir::LogisticKernel(10.0))},
                                            {42, sparseir::compute_sve<sparseir::LogisticKernel, double>(sparseir::LogisticKernel(42.0))},
                                            {10000, sparseir::compute_sve<sparseir::LogisticKernel, double>(sparseir::LogisticKernel(10000.0))},
                                            {100000000, sparseir::compute_sve<sparseir::LogisticKernel, double>(sparseir::LogisticKernel(10000.0), 1e-12)}};

    SECTION("smooth with Λ =") {
        for (int Lambda : {10, 42, 10000}) {
            // sparseir::FiniteTempBasis<sparseir::Fermionic, sparseir::LogisticKernel> basis(1, Lambda, sve_logistic[Lambda]);
            // TODO: Check that the maximum implementation  is defined
            // check_smooth(basis.u, basis.s, 2 * sparseir::maximum(basis.u(1)), 24);
            //check_smooth(basis.v, basis.s, 50, 200);
        }
    }

    /*
    SECTION("num roots u with Λ =") {
        for (int Lambda : {10, 42, 10000}) {
            FiniteTempBasis<Fermionic> basis(1, Lambda, sparseir::sve_logistic[Lambda]);
            for (const auto& ui : basis.u) {
                std::vector<double> ui_roots = sparseir::roots(ui);
                REQUIRE(ui_roots.size() == static_cast<size_t>(ui.l));
            }
        }
    }
    */

    /*
    SECTION("num roots û with stat =, Λ =") {
        for (const auto& stat : {Fermionic(), Bosonic()}) {
            for (int Lambda : {10, 42, 10000}) {
                FiniteTempBasis basis(stat, 1, Lambda, sparseir::sve_logistic[Lambda]);
                for (int i : {1, 2, 8, 11}) {
                    std::vector<double> x0 = sparseir::find_extrema(basis.uhat[i]);
                    REQUIRE(i <= x0.size() && x0.size() <= static_cast<size_t>(i + 1));
                }
            }
        }
    }
    */

    /*
    SECTION("accuracy with stat =, Λ =") {
        for (const auto& stat : {Fermionic(), Bosonic()}) {
            for (int Lambda : {10, 42, 10000}) {
                FiniteTempBasis basis(stat, 4, Lambda, sparseir::sve_logistic[Lambda]);
                REQUIRE(sparseir::accuracy(basis) <= sparseir::significance(basis).back());
                REQUIRE(sparseir::significance(basis).front() == 1.0);
                REQUIRE(sparseir::accuracy(basis) <= (basis.s.back() / basis.s.front()));
            }
        }
    }
    */

    /*
    SECTION("choose_accuracy") {
        // Suppress warnings using logger
        sparseir::with_logger(sparseir::NullLogger(), [&]() {
            REQUIRE(sparseir::choose_accuracy({}, {}) == std::make_tuple(2.2204460492503131e-16, Float64x2(), "default"));
            REQUIRE(sparseir::choose_accuracy({}, Float64()) == std::make_tuple(1.4901161193847656e-8, Float64(), "default"));
            REQUIRE(sparseir::choose_accuracy({}, Float64x2()) == std::make_tuple(2.2204460492503131e-16, Float64x2(), "default"));
            REQUIRE(sparseir::choose_accuracy(1e-6, {}) == std::make_tuple(1.0e-6, Float64(), "default"));
            REQUIRE(sparseir::choose_accuracy(1e-8, {}) == std::make_tuple(1.0e-8, Float64x2(), "default"));

            REQUIRE(sparseir::choose_accuracy(1e-20, {}) == std::make_tuple(1.0e-20, Float64x2(), "default"));
            // Note: Catch2 doesn't have a built-in way to capture logs.
            // You might need to implement a custom logger or use a library that supports log capturing.

            REQUIRE(sparseir::choose_accuracy(1e-10, Float64()) == std::make_tuple(1.0e-10, Float64(), "accurate"));

            REQUIRE(sparseir::choose_accuracy(1e-6, Float64()) == std::make_tuple(1.0e-6, Float64(), "default"));
            REQUIRE(sparseir::choose_accuracy(1e-6, Float64(), "auto") == std::make_tuple(1.0e-6, Float64(), "default"));
            REQUIRE(sparseir::choose_accuracy(1e-6, Float64(), "accurate") == std::make_tuple(1.0e-6, Float64(), "accurate"));
        });
    }
    */

    /*
    SECTION("truncate") {
        sparseir::CentrosymmSVE sve(LogisticKernel(5), 1e-6, Float64());

        auto svds = sparseir::compute_svd(sparseir::matrices(sve));
        std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> svd_tuple = svds;
        std::vector<double> u_ = std::get<0>(svd_tuple);
        std::vector<double> s_ = std::get<1>(svd_tuple);
        std::vector<double> v_ = std::get<2>(svd_tuple);

        for (int lmax = 3; lmax <= 20; ++lmax) {
            std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> truncated = sparseir::truncate(u_, s_, v_, lmax);
            std::vector<double> u = std::get<0>(truncated);
            std::vector<double> s = std::get<1>(truncated);
            std::vector<double> v = std::get<2>(truncated);

            std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> postprocessed = sparseir::postprocess(sve, u, s, v);
            std::vector<double> u_post = std::get<0>(postprocessed);
            std::vector<double> s_post = std::get<1>(postprocessed);
            std::vector<double> v_post = std::get<2>(postprocessed);
            REQUIRE(u_post.size() == s_post.size());
            REQUIRE(s_post.size() == v_post.size());
            REQUIRE(s_post.size() <= static_cast<size_t>(lmax - 1));
        }
    }
    */
}