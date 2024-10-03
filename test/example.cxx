#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "sparseir/sparseir.h"
#include <xprec/ddouble.h>
#include <Eigen/Core>
#include <Eigen/SVD>

using namespace xprec;

uint32_t factorial(uint32_t number)
{
    return number <= 1 ? number : factorial(number - 1) * number;
}

TEST_CASE("Factorials are computed", "[factorial]")
{
    REQUIRE(factorial(1) == 1);
    REQUIRE(factorial(2) == 2);
    REQUIRE(factorial(3) == 6);
    REQUIRE(factorial(10) == 3'628'800);
}

TEST_CASE(" test ", "[test]")
{
    using MatrixXdd = Eigen::Matrix<DDouble, Eigen::Dynamic, Eigen::Dynamic>;
    auto N = 20;
    auto m = MatrixXdd(N, N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            m(i, j) = 1 / double(i + j + 1);
        }
    }
    Eigen::JacobiSVD<MatrixXdd> svd;
    svd.compute(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
}