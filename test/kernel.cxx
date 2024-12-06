#include <catch2/catch_test_macros.hpp>
#include <Eigen/Core>
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <iostream>
#include <memory>

// Include your sparseir headers
#include <sparseir/sparseir-header-only.hpp>

using xprec::DDouble;

template<typename Kernel>
bool kernel_accuracy_test(Kernel &K) {
    using T = float;
    using T_x = double;

    // Convert Rule to type T
    auto ddouble_rule = sparseir::legendre(10);
    auto double_rule = sparseir::convert<double>(ddouble_rule);
    auto rule = sparseir::convert<T>(double_rule);

    // Obtain SVE hints for the kernel
    auto hints = sparseir::sve_hints(K, 2.2e-16);

    // Generate piecewise Gaussian quadrature rules for x and y

    auto sx = hints.segments_x();
    auto sy = hints.segments_y();

    REQUIRE(std::is_sorted(sx.begin(), sx.end()));
    REQUIRE(std::is_sorted(sy.begin(), sy.end()));

    auto gauss_x = rule.piecewise(sx);
    auto gauss_y = rule.piecewise(sy);

    T epsilon = std::numeric_limits<T>::epsilon();
    T tiny = std::numeric_limits<T>::min() / epsilon;

    // Compute the matrix from Gaussian quadrature
    auto result = sparseir::matrix_from_gauss<T>(K, gauss_x, gauss_y);

    // Convert gauss_x and gauss_y to higher precision T_x
    auto gauss_x_Tx = sparseir::convert<T_x>(gauss_x);
    auto gauss_y_Tx = sparseir::convert<T_x>(gauss_y);

    // Compute the matrix in higher precision
    auto result_x = sparseir::matrix_from_gauss<T_x>(K, gauss_x_Tx, gauss_y_Tx);
    T_x magn = result_x.cwiseAbs().maxCoeff();

    // Check that the difference is within tolerance
    REQUIRE((result.template cast<T_x>() - result_x).cwiseAbs().maxCoeff() <= 2 * magn * epsilon);

    auto reldiff = (result.cwiseAbs().array() < tiny)
                       .select(T(1.0), result.array() / result_x.template cast<T>().array());

    REQUIRE((reldiff - T(1.0)).cwiseAbs().maxCoeff() <= 100 * epsilon);
    return true;
}

TEST_CASE("Kernel Accuracy Test")
{
    {
        // List of kernels to test
        std::vector<sparseir::LogisticKernel> kernels = {
            sparseir::LogisticKernel(9.0),
            //sparseir::RegularizedBoseKernel(8.0),
            sparseir::LogisticKernel(120000.0),
            //sparseir::RegularizedBoseKernel(127500.0),
            // Symmetrized kernels
        };
        for (const auto &K : kernels)
        {
            REQUIRE(kernel_accuracy_test(K));
        }
    }
    {
        auto kernel_ptr = std::make_shared<const sparseir::LogisticKernel>(40000.0);
        auto k1 = sparseir::get_symmetrized(kernel_ptr, -1);
        // TODO: implement sve_hints
        // REQUIRE(kernel_accuracy_test(k1));
        //auto k2 = sparseir::get_symmetrized(sparseir::RegularizedBoseKernel(40000.0), -1);
        // TODO: implement sve_hints
        // REQUIRE(kernel_accuracy_test(k2));
    }
    {
        // List of kernels to test
        std::vector<sparseir::RegularizedBoseKernel> kernels = {
            sparseir::RegularizedBoseKernel(8.0),
            sparseir::RegularizedBoseKernel(127500.0),
        };
        for (const auto &K : kernels)
        {
            REQUIRE(kernel_accuracy_test(K));
        }
    }
    /*
    {
        // List of kernels to test
        std::vector<sparseir::RegularizedBoseKernel> kernels = {
            // Symmetrized kernels
            sparseir::LogisticKernel(40000.0).get_symmetrized(-1),
            sparseir::RegularizedBoseKernel(35000.0).get_symmetrized(-1),
        };
        for (const auto &K : kernels)
        {
            REQUIRE(kernel_accuracy_test(K));
        }
    }
    */
}

TEST_CASE("Kernel Singularity Test")
{
    std::vector<double> lambdas = {10.0, 42.0, 10000.0};
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (double lambda : lambdas)
    {
        // Generate random x values
        std::vector<double> x_values(1000);
        for (double &x : x_values)
        {
            x = dist(rng);
        }

        sparseir::RegularizedBoseKernel K(lambda);

        for (double x : x_values)
        {
            double expected = 1.0 / lambda;
            double computed = K(x, 0.0);
            REQUIRE(Eigen::internal::isApprox(computed, expected));
        }
    }
}