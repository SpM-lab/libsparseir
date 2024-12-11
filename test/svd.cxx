#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

// test_piecewise_legendre_poly.cpp

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

TEST_CASE("svd.cpp")
{
    using namespace sparseir;
    using namespace Eigen;
    using namespace xprec;

    // Create a matrix of Float64x2 equivalent (here just Eigen::MatrixXd for
    // simplicity)
    /**
    Eigen::MatrixXd mat64x2 = Eigen::MatrixXd::Random(4, 6);
    REQUIRE_NOTHROW(sparseir::compute_svd(mat64x2, "accurate", 2));
    std::cout << "n_sv_hint is set but will not be used in the current
    implementation!" << std::endl;

    REQUIRE_NOTHROW({
        compute_svd(mat64x2, "accurate");
        std::cout << "strategy is set but will not be used in the current
    implementation!" << std::endl;
    });
    */

    // Create a standard matrix
    MatrixX<DDouble> mat = MatrixX<DDouble>::Random(5, 6);
    auto svd_result = compute_svd(mat, 0, "default");
    auto U = std::get<0>(svd_result);
    auto S = std::get<1>(svd_result);
    auto V = std::get<2>(svd_result);
    auto diff = (mat - U * S.asDiagonal() * V.transpose()).norm() / mat.norm();
    REQUIRE(diff < 1e-28);

    /*
    REQUIRE_THROWS_AS(compute_svd(mat, "fast"), std::domain_error);
    */
}