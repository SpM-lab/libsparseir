#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>
#include <catch2/catch_approx.hpp>

#include <sparseir/sparseir-header-only.hpp>

using Catch::Approx;
using xprec::DDouble;

template <typename Kernel>
std::tuple<bool, bool> kernel_accuracy_test(Kernel &K)
{
    using T = float;
    using T_x = double;

    // Convert Rule to type T
    auto ddouble_rule = sparseir::legendre(10);
    auto double_rule = sparseir::Rule<double>(ddouble_rule);
    auto rule = sparseir::Rule<T>(double_rule);

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
    auto gauss_x_Tx = sparseir::Rule<T_x>(gauss_x);
    auto gauss_y_Tx = sparseir::Rule<T_x>(gauss_y);

    // Compute the matrix in higher precision
    auto result_x = sparseir::matrix_from_gauss<T_x>(K, gauss_x_Tx, gauss_y_Tx);
    T_x magn = result_x.cwiseAbs().maxCoeff();

    // Check that the difference is within tolerance
    bool b1 = (result.template cast<T_x>() - result_x).cwiseAbs().maxCoeff() <=
              2 * magn * epsilon;

    auto reldiff = (result.cwiseAbs().array() < tiny)
                       .select(T(1.0), result.array() /
                                           result_x.template cast<T>().array());

    bool b2 = (reldiff - T(1.0)).cwiseAbs().maxCoeff() <= 100 * epsilon;
    return std::make_tuple(b1, b2);
}

TEST_CASE("LogisticKernel", "[kernel]")
{
    auto kernel = sparseir::LogisticKernel(40000.0);
    auto reduced_kernel = sparseir::get_symmetrized(kernel, std::integral_constant<int, +1>{});
    REQUIRE(true);
}

//TEST_CASE("Kernel Accuracy Test")
//{
    //using T = double;
    //{
        //// List of kernels to test
        //std::vector<std::shared_ptr<const sparseir::LogisticKernel<T>>> kernels = {
            //std::make_shared<const sparseir::LogisticKernel<T>>(9.0),
            // sparseir::RegularizedBoseKernel(8.0),
            //std::make_shared<const sparseir::LogisticKernel<T>>(120000.0),
            //// sparseir::RegularizedBoseKernel(127500.0),
            ////  Symmetrized kernels
        //};
        //for (const auto K : kernels) {
            //bool b1, b2;
            //std::tie(b1, b2) = kernel_accuracy_test(*K);
            //REQUIRE(b1);
            //REQUIRE(b2);
            ///*
            //if (b1){
                //std::cout << "Kernel accuracy test passed for " <<
            //typeid(K).name() << b1 << b2 << std::endl;
            //}
//
            //if (b2){
                //std::cout << "Kernel accuracy test passed for " <<
            //typeid(K).name() << b1 << b2 << std::endl;
            //}
            //*/
        //}
    //}
    //{
        //auto kernel_ptr =
            //std::make_shared<const sparseir::LogisticKernel<T>>(40000.0);
        //auto K = sparseir::get_symmetrized(
            //std::static_pointer_cast<const sparseir::AbstractKernel<T>>(kernel_ptr), -1);
        //// TODO: implement sve_hints
        //bool b1, b2;
        // std::tie(b1, b2) = kernel_accuracy_test(K);
        // REQUIRE(b1);
        // REQUIRE(b2);
        //  TODO: resolve this errors
        // auto k2 =
        // sparseir::get_symmetrized(sparseir::RegularizedBoseKernel(40000.0),
        // -1);
        //  TODO: implement sve_hints
        //  REQUIRE(kernel_accuracy_test(k2));
    //}
    //{
        //// List of kernels to test
        //std::vector<std::shared_ptr<sparseir::RegularizedBoseKernel<T>>> kernels = {
            //std::make_shared<sparseir::RegularizedBoseKernel<T>>(8.0),
            //std::make_shared<sparseir::RegularizedBoseKernel<T>>(127500.0),
        //};
        //for (const auto K : kernels) {
            //bool b1, b2;
            //std::tie(b1, b2) = kernel_accuracy_test(*K);
            //// TODO: resolve this errors
            //REQUIRE(b1);
            //// TODO: resolve this errors
            //REQUIRE(b2);
            ///*
            //if (b1)
            //{
                //std::cout << "Kernel accuracy test passed for " <<
            //typeid(K).name() << b1 << b2 << std::endl;
            //}
//
            //if (b2)
            //{
                //std::cout << "Kernel accuracy test passed for " <<
            //typeid(K).name() << b1 << b2 << std::endl;
            //}
            //*/
        //}
    //}
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
//}

TEST_CASE("weight_func for LogisticKernel(42)", "[kernel]")
{
    sparseir::LogisticKernel K(42);

    std::integral_constant<int, +1> plus_one{};
    std::integral_constant<int, -1> minus_one{};
    sparseir::ReducedKernel<sparseir::LogisticKernel> K_symm =
        sparseir::get_symmetrized(K, plus_one);
    REQUIRE(!K_symm.is_centrosymmetric());

    {
        auto weight_func_bosonic = K.weight_func<double>(sparseir::Bosonic());
        REQUIRE(weight_func_bosonic(1e-16) == 1.0 / tanh(0.5 * 42 * 1e-16));

        auto weight_func_fermionic =
            K.weight_func<double>(sparseir::Fermionic());
        REQUIRE(weight_func_fermionic(482) == 1.0);

        auto weight_func_symm_bosonic =
            K_symm.weight_func<double>(sparseir::Bosonic());
        REQUIRE(weight_func_symm_bosonic(1e-16) == 1.0);

        auto weight_func_symm_fermionic =
            K_symm.weight_func<double>(sparseir::Fermionic());
        REQUIRE(weight_func_symm_fermionic(482) == 1.0);
    }

    {
        auto weight_func_bosonic =
            K.weight_func<xprec::DDouble>(sparseir::Bosonic());
        REQUIRE(weight_func_bosonic(1e-16) ==
                Approx(1.0 / std::tanh(0.5 * 42 * 1e-16)));

        auto weight_func_fermionic =
            K.weight_func<xprec::DDouble>(sparseir::Fermionic());
        REQUIRE(weight_func_fermionic(482) == 1.0);

        auto weight_func_symm_bosonic =
            K_symm.weight_func<xprec::DDouble>(sparseir::Bosonic());
        REQUIRE(weight_func_symm_bosonic(1e-16) == 1.0);

        auto weight_func_symm_fermionic =
            K_symm.weight_func<xprec::DDouble>(sparseir::Fermionic());
        REQUIRE(weight_func_symm_fermionic(482) == 1.0);
    }
}

TEST_CASE("kernel.cpp", "[kernel]")
{
    SECTION("even")
    {
        double epsilon = 2.220446049250313e-16;
        auto kernel = sparseir::LogisticKernel(5.0);
        auto reduced_kernel = sparseir::get_symmetrized(
            kernel, std::integral_constant<int, +1>{});
        REQUIRE(reduced_kernel.inner.lambda_ == kernel.lambda_);
        REQUIRE(reduced_kernel.sign == +1);
    }

    SECTION("odd")
    {
        double epsilon = 2.220446049250313e-16;
        auto kernel = sparseir::LogisticKernel(5.0);
        auto reduced_kernel = sparseir::get_symmetrized(
            kernel, std::integral_constant<int, -1>{});
        REQUIRE(reduced_kernel.inner.lambda_ == kernel.lambda_);
        REQUIRE(reduced_kernel.sign == -1);
    }
}

TEST_CASE("Kernel Singularity Test", "[kernel]")
{
    using T = xprec::DDouble;
    std::vector<double> lambdas = {10.0, 42.0, 10000.0};
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (double lambda : lambdas) {
        // Generate random x values
        std::vector<double> x_values(1000);
        for (double &x : x_values) {
            x = dist(rng);
        }

        auto K = sparseir::RegularizedBoseKernel(lambda);

        for (double x : x_values) {
            T expected = 1.0 / static_cast<T>(lambda);
            T computed = K.compute(static_cast<T>(x), static_cast<T>(0.0));
            REQUIRE(Eigen::internal::isApprox(computed, expected));
        }
    }
}