#include <catch2/catch_test_macros.hpp>
#include <Eigen/Core>
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <memory>

// Include your sparseir headers
#include <sparseir/sparseir-header-only.hpp>

using xprec::DDouble;

TEST_CASE("Kernel Accuracy Test")
{
    using T = float;
    using T_x = double;

    // List of kernels to test
    std::vector<std::shared_ptr<sparseir::AbstractKernel>> kernels = {
        std::make_shared<sparseir::LogisticKernel>(9.0),
        std::make_shared<sparseir::RegularizedBoseKernel>(8.0),
        std::make_shared<sparseir::LogisticKernel>(120000.0),
        std::make_shared<sparseir::RegularizedBoseKernel>(127500.0),
        // Symmetrized kernels
        std::make_shared<sparseir::LogisticKernel>(40000.0)->get_symmetrized(-1),
        std::make_shared<sparseir::RegularizedBoseKernel>(35000.0)->get_symmetrized(-1),
    };

    for (const auto &K_ptr : kernels)
    {
        const auto &K = *K_ptr;

        // Convert Rule to type T
        sparseir::Rule<DDouble> _rule = sparseir::legendre(10);
        sparseir::Rule<T> rule = sparseir::convert<T>(_rule);

        // Obtain SVE hints for the kernel
        auto hints = K.sve_hints(2.2e-16);

        // Generate piecewise Gaussian quadrature rules for x and y
        auto gauss_x = sparseir::piecewise(rule, hints.template segments_x<T>());
        auto gauss_y = sparseir::piecewise(rule, hints.template segments_y<T>());

        T epsilon = std::numeric_limits<T>::epsilon();
        T tiny = std::numeric_limits<T>::min() / epsilon;

        // Compute the matrix from Gaussian quadrature
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result = sparseir::matrix_from_gauss(K, gauss_x, gauss_y);

        // Convert gauss_x and gauss_y to higher precision T_x
        auto gauss_x_Tx = sparseir::convert<T_x>(gauss_x);
        auto gauss_y_Tx = sparseir::convert<T_x>(gauss_y);

        // Compute the matrix in higher precision
        Eigen::Matrix<T_x, Eigen::Dynamic, Eigen::Dynamic> result_x = sparseir::matrix_from_gauss(K, gauss_x_Tx, gauss_y_Tx);

        T_x magn = result_x.cwiseAbs().maxCoeff();

        // Check that the difference is within tolerance
        REQUIRE((result.template cast<T_x>() - result_x).cwiseAbs().maxCoeff() <= 2 * magn * epsilon);

        Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> reldiff = (result.cwiseAbs().array() < tiny)
                                                                      .select(T(1.0), result.array() / result_x.template cast<T>().array());

        REQUIRE((reldiff - T(1.0)).cwiseAbs().maxCoeff() <= 100 * epsilon);
    }
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
            REQUIRE(isApprox(computed, expected));
        }
    }
}