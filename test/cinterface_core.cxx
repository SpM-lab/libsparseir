#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <sparseir/sparseir.h>   // C interface
#include <sparseir/sparseir.hpp> // C++ interface

using Catch::Approx;
using xprec::DDouble;

TEST_CASE("Kernel Accuracy Tests", "[cinterface]")
{
    // Test individual kernels
    SECTION("LogisticKernel(9)")
    {
        auto cpp_kernel = sparseir::LogisticKernel(9);
        int status;
        spir_kernel *kernel = spir_logistic_kernel_new(9, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(kernel != nullptr);
    }

    SECTION("RegularizedBoseKernel(10)")
    {
        auto cpp_kernel = sparseir::RegularizedBoseKernel(10);
        int status;
        spir_kernel *kernel = spir_regularized_bose_kernel_new(10, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(kernel != nullptr);
    }

    SECTION("Kernel Domain")
    {
        // Create a kernel through C API
        int kernel_status;
        spir_kernel *kernel = spir_logistic_kernel_new(9, &kernel_status);
        REQUIRE(kernel_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(kernel != nullptr);

        // Get domain bounds
        double xmin, xmax, ymin, ymax;
        int domain_status = spir_kernel_domain(kernel, &xmin, &xmax, &ymin, &ymax);
        REQUIRE(domain_status == SPIR_COMPUTATION_SUCCESS);

        // Compare with C++ implementation
        auto cpp_kernel = sparseir::LogisticKernel(9);
        auto xrange = cpp_kernel.xrange();
        auto yrange = cpp_kernel.yrange();
        auto cpp_xmin = xrange.first;
        auto cpp_xmax = xrange.second;
        auto cpp_ymin = yrange.first;
        auto cpp_ymax = yrange.second;

        REQUIRE(xmin == cpp_xmin);
        REQUIRE(xmax == cpp_xmax);
        REQUIRE(ymin == cpp_ymin);
        REQUIRE(ymax == cpp_ymax);

        // Clean up
        spir_kernel_destroy(kernel);
    }
}

template <typename S>
spir_statistics_type get_stat()
{
    if (std::is_same<S, sparseir::Fermionic>::value) {
        return SPIR_STATISTICS_FERMIONIC;
    } else {
        return SPIR_STATISTICS_BOSONIC;
    }
}

template <typename S>
void test_finite_temp_basis_constructor()
{
    double beta = 2.0;
    double wmax = 5.0;
    double Lambda = 10.0;
    double epsilon = 1e-6;

    auto stat = get_stat<S>();

    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, epsilon);
    int basis_status;
    spir_basis *basis = spir_basis_new(stat, beta, wmax, epsilon, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    int basis_size;
    int size_status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(size_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis_size == cpp_basis.size());
}

template <typename S>
void test_finite_temp_basis_constructor_with_sve()
{
    double beta = 2.0;
    double wmax = 5.0;
    double Lambda = 10.0;
    double epsilon = 1e-6;

    int kernel_status;
    spir_kernel *kernel = spir_logistic_kernel_new(Lambda, &kernel_status);
    REQUIRE(kernel_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);

    int sve_status;
    spir_sve_result *sve_result = spir_sve_result_new(kernel, epsilon, &sve_status);
    REQUIRE(sve_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sve_result != nullptr);

    auto stat = get_stat<S>();

    int basis_status;
    spir_basis *basis = spir_basis_new_with_sve(
        stat, beta, wmax, kernel, sve_result, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    spir_statistics_type stats;
    int stats_status = spir_basis_get_statistics(basis, &stats);
    REQUIRE(stats_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(stats == stat);

    // Clean up
    spir_kernel_destroy(kernel);
    spir_sve_result_destroy(sve_result);
    spir_basis_destroy(basis);
}

TEST_CASE("FiniteTempBasis", "[cinterface]")
{
    SECTION("FiniteTempBasis Constructor Fermionic")
    {
        test_finite_temp_basis_constructor<sparseir::Fermionic>();
    }

    SECTION("FiniteTempBasis Constructor Bosonic")
    {
        test_finite_temp_basis_constructor<sparseir::Bosonic>();
    }

    SECTION("FiniteTempBasis Constructor with SVE Fermionic/LogisticKernel")
    {
        test_finite_temp_basis_constructor_with_sve<sparseir::Fermionic>();
    }

    SECTION("FiniteTempBasis Constructor with SVE Bosonic/LogisticKernel")
    {
        test_finite_temp_basis_constructor_with_sve<sparseir::Bosonic>();
    }

    SECTION(
        "FiniteTempBasis Constructor with SVE Bosonic/RegularizedBoseKernel")
    {
        double beta = 2.0;
        double wmax = 5.0;
        double Lambda = 10.0;
        double epsilon = 1e-6;

        int kernel_status;
        spir_kernel *kernel = spir_regularized_bose_kernel_new(Lambda, &kernel_status);
        REQUIRE(kernel_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(kernel != nullptr);

        int sve_status;
        spir_sve_result *sve_result = spir_sve_result_new(kernel, epsilon, &sve_status);
        REQUIRE(sve_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(sve_result != nullptr);

        int basis_status;
        spir_basis *basis = spir_basis_new_with_sve(
            SPIR_STATISTICS_BOSONIC, beta, wmax, kernel, sve_result, &basis_status);
        REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(basis != nullptr);

        spir_statistics_type stats;
        int stats_status = spir_basis_get_statistics(basis, &stats);
        REQUIRE(stats_status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(stats == SPIR_STATISTICS_BOSONIC);

        // Clean up
        spir_kernel_destroy(kernel);
        spir_sve_result_destroy(sve_result);
        spir_basis_destroy(basis);
    }
}

template <typename S>
void test_finite_temp_basis_basis_functions()
{
    double beta = 2.0;
    double wmax = 5.0;
    double epsilon = 1e-6;

    auto stat = get_stat<S>();

    int basis_status;
    spir_basis *basis = spir_basis_new(stat, beta, wmax, epsilon, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    int u_status;
    spir_funcs *u = spir_basis_get_u(basis, &u_status);
    REQUIRE(u_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(u != nullptr);

    int uhat_status;
    spir_funcs *uhat = spir_basis_get_uhat(basis, &uhat_status);
    REQUIRE(uhat_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(uhat != nullptr);

    // Test basis function evaluation
    int basis_size;
    int size_status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(size_status == SPIR_COMPUTATION_SUCCESS);

    double x = 0.5;        // Test point for u basis (imaginary time)
    double y = 0.5 * wmax; // Test point for v basis (real frequency)
    double *out = (double *)malloc(basis_size * sizeof(double));
    int eval_status = spir_funcs_evaluate(u, x, out);
    REQUIRE(eval_status == SPIR_COMPUTATION_SUCCESS);

    // Compare with C++ implementation for u basis
    sparseir::FiniteTempBasis<S> cpp_basis(beta, wmax, epsilon);
    Eigen::VectorXd cpp_result = (*cpp_basis.u)(x);
    for (int i = 0; i < basis_size; ++i) {
        REQUIRE(out[i] == Approx(cpp_result(i)));
    }

    // Test v basis functions
    int v_status;
    spir_funcs *v = spir_basis_get_v(basis, &v_status);
    REQUIRE(v_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(v != nullptr);

    // Test v basis function evaluation
    eval_status = spir_funcs_evaluate(v, y, out);
    REQUIRE(eval_status == SPIR_COMPUTATION_SUCCESS);

    // Compare with C++ implementation for v basis
    cpp_result = (*cpp_basis.v)(y);
    for (int i = 0; i < basis_size; ++i) {
        REQUIRE(out[i] == Approx(cpp_result(i)));
    }

    free(out);
    spir_funcs_destroy(u);
    spir_funcs_destroy(v);
    spir_funcs_destroy(uhat);

    // Test error cases
    eval_status = spir_funcs_evaluate(nullptr, x, out);
    REQUIRE(eval_status == SPIR_INVALID_ARGUMENT);

    eval_status = spir_funcs_evaluate(u, x, nullptr);
    REQUIRE(eval_status == SPIR_INVALID_ARGUMENT);

    // Clean up
    spir_basis_destroy(basis);
}

TEST_CASE("FiniteTempBasis Basis Functions", "[cinterface]")
{
    SECTION("Basis Functions Fermionic")
    {
        test_finite_temp_basis_basis_functions<sparseir::Fermionic>();
    }

    SECTION("Basis Functions Bosonic")
    {
        test_finite_temp_basis_basis_functions<sparseir::Bosonic>();
    }
}