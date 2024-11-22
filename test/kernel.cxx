#include <Eigen/Core>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>

using xprec::DDouble;

TEST_CASE("kernel.hpp")
{
    // Use a vector of shared pointers to AbstractKernel objects
    std::vector<std::shared_ptr<sparseir::AbstractKernel>> kernels = {
        std::make_shared<sparseir::LogisticKernel>(9.0),
        std::make_shared<sparseir::RegularizedBoseKernel>(8.0),
        std::make_shared<sparseir::LogisticKernel>(120000.0),
        std::make_shared<sparseir::RegularizedBoseKernel>(127500.0),
        // Obtain symmetrized kernels
        //sparseir::get_symmetrized(std::make_shared<sparseir::LogisticKernel>(40000.0), -1),
        //sparseir::get_symmetrized(std::make_shared<sparseir::RegularizedBoseKernel>(35000.0), -1)
    };

    for (const auto& K_ptr : kernels)
    {
        const auto& K = *K_ptr; // Dereference the shared_ptr to get the kernel

        using T = float;
        using T_x = double;

        // Convert Rule to type T
        sparseir::Rule<DDouble> _rule = sparseir::legendre(10);
        sparseir::Rule<T> rule = sparseir::convert<T>(_rule);

        // Obtain SVE hints for the kernel
        // TODO: Implement sve_hints
        // auto hints = sparseir::sve_hints(K, std::numeric_limits<T_x>::epsilon());

        // Generate piecewise Gaussian quadrature rules for x and y
        // TODO: Implement piecewise and segments_x and segments_y
        // auto gauss_x = sparseir::piecewise(rule, sparseir::segments_x(hints));
        // auto gauss_y = sparseir::piecewise(rule, sparseir::segments_y(hints));

        T epsilon = std::numeric_limits<T>::epsilon();
        T tiny = std::numeric_limits<T>::min() / epsilon;

        // Compute the matrix from Gaussian quadrature
        // TODO: Implement matrix_from_gauss
        // Eigen::MatrixX<T> result = sparseir::matrix_from_gauss(K, gauss_x, gauss_y);

        // Convert gauss_x and gauss_y to higher precision T_x
        // TODO: Implement convert
        // auto gauss_x_Tx = sparseir::convert<T_x>(gauss_x);
        // auto gauss_y_Tx = sparseir::convert<T_x>(gauss_y);

        // Compute the matrix in higher precision
        // TODO: Implement matrix_from_gauss
        // Eigen::MatrixX<T_x> result_x = sparseir::matrix_from_gauss(K, gauss_x_Tx, gauss_y_Tx);

        // T_x magn = result_x.cwiseAbs().maxCoeff();

        // Check that the difference is within tolerance
        // REQUIRE((result.template cast<T_x>() - result_x).cwiseAbs().maxCoeff() <= 2 * magn * epsilon);

        // Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> reldiff = (result.cwiseAbs().array() < tiny)
        //    .select(T(1.0), result.array() / result_x.template cast<T>().array());

        // REQUIRE((reldiff - T(1.0)).cwiseAbs().maxCoeff() <= 100 * epsilon);
    }

    // Second test set: "singularity with Λ = Λ"
    std::vector<double> lambdas = {10.0, 42.0, 10000.0};
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (double Lambda : lambdas)
    {
        std::vector<double> xs(10);
        for (auto& x : xs)
        {
            x = dist(rng);
        }

        auto K_ptr = std::make_shared<sparseir::RegularizedBoseKernel>(Lambda);
        const auto& K = *K_ptr;
        const double tol = 1e-6;

        for (double x : xs)
        {
            REQUIRE(std::abs(K(x, 0.0) - 1.0 / Lambda) <= tol);
        }
    }

    // Third test set: "unit tests"
    {
        auto K_ptr = std::make_shared<sparseir::LogisticKernel>(42.0);
        const auto& K = *K_ptr;
        auto K_symm_ptr = sparseir::get_symmetrized(K_ptr, 1);
        const auto& K_symm = *K_symm_ptr;

        REQUIRE(!sparseir::iscentrosymmetric(K_symm));

        REQUIRE_THROWS_AS(sparseir::get_symmetrized(K_symm_ptr, -1), std::runtime_error);

        double expected_value = 1.0 / std::tanh(0.5 * 42.0 * 1e-8);
        double computed_value = sparseir::weight_func(K, sparseir::Bosonic())(1e-8);
        REQUIRE(computed_value == Approx(expected_value));

        REQUIRE(sparseir::weight_func(K, sparseir::Fermionic())(482) == Approx(1.0));

        REQUIRE(sparseir::weight_func(K_symm, sparseir::Bosonic())(1e-3) == Approx(1.0));

        REQUIRE(sparseir::weight_func(K_symm, sparseir::Fermionic())(482) == Approx(1.0));

        auto K99_ptr = std::make_shared<sparseir::RegularizedBoseKernel>(99.0);
        const auto& K99 = *K99_ptr;
        auto hints = sparseir::sve_hints(K99, 1e-6);

        REQUIRE(sparseir::nsvals(hints) == 56);

        REQUIRE(sparseir::ngauss(hints) == 10);

        REQUIRE(sparseir::ypower(K99) == 1);

        REQUIRE(sparseir::ypower(*sparseir::get_symmetrized(K99_ptr, -1)) == 1);

        REQUIRE(sparseir::ypower(*sparseir::get_symmetrized(K99_ptr, 1)) == 1);

        REQUIRE(sparseir::conv_radius(K99) == Approx(40.0 * 99.0));

        REQUIRE(sparseir::conv_radius(*sparseir::get_symmetrized(K99_ptr, -1)) == Approx(40.0 * 99.0));

        REQUIRE(sparseir::conv_radius(*sparseir::get_symmetrized(K99_ptr, 1)) == Approx(40.0 * 99.0));

        REQUIRE(sparseir::weight_func(K99, sparseir::Bosonic())(482) == Approx(1.0 / 482.0));

        REQUIRE_THROWS_AS(sparseir::weight_func(K99, sparseir::Fermionic())(1.0), std::runtime_error);
    }
}