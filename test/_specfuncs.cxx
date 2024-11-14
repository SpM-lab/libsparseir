#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

#include <xprec/ddouble-header-only.hpp>

#include <sparseir/_specfuncs.hpp>

using xprec::DDouble;
using Eigen::MatrixXd;

TEST_CASE("_specfuns.cxx"){
    SECTION("legval"){
        std::vector<double> c = {1.0, 2.0, 3.0};
        double x = 0.5;
        double result = sparseir::legval(x, c);
        REQUIRE(result == 1.625);
    }

    SECTION("legvander"){
        std::vector<double> x = {0.0, 0.5, 1.0};
        int deg = 2;
        MatrixXd result = sparseir::legvander(x, deg);
        MatrixXd expected(3, 3);
        expected << 1, 0, -0.5, 1.0, 0.5, -0.125, 1, 1, 1;
        REQUIRE(result.isApprox(expected, 1e-9));
    }

}
