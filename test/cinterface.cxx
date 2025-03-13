#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>
#include <catch2/catch_approx.hpp>

#include <sparseir/sparseir.h> // C interface
#include <sparseir/sparseir.hpp> // C++ interface

using Catch::Approx;
using xprec::DDouble;

TEST_CASE("Kernel Accuracy Tests", "[cinterface]")
{
    // Test individual kernels
    SECTION("LogisticKernel(9)")
    {
        auto kernel = sparseir::LogisticKernel(9);
    }

    SECTION("Kernel Domain")
    {
        // Create a kernel through C API
        //spir_logistic_kernel* kernel = spir_logistic_kernel_new(9);
        spir_kernel* kernel = spir_logistic_kernel_new(9);
        REQUIRE(kernel != nullptr);

        // Get domain bounds
        double xmin, xmax, ymin, ymax;
        int status = spir_kernel_domain(kernel, &xmin, &xmax, &ymin, &ymax);
        REQUIRE(status == 0);

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
        spir_destroy_kernel(kernel);
    }
}

TEST_CASE("TauSampling", "[cinterface]")
{
    SECTION("TauSampling Constructor")
    {
        double beta = 1.0;
        double wmax = 10.0;

        auto basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-15);
        REQUIRE(basis != nullptr);

        auto sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);
        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
    }

    SECTION("TauSampling Evaluation 1-dimensional input")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis* basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling* sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(
            beta, wmax, 1e-10, sparseir::LogisticKernel(beta * wmax));
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();
        std::cout << "basis_size: " << basis_size << std::endl;
        Eigen::VectorXd cpp_Gl_vec = Eigen::VectorXd::Random(basis_size);
        Eigen::Tensor<double, 1> cpp_Gl(basis_size);
        for (size_t i = 0; i < basis_size; ++i) {
            cpp_Gl(i) = cpp_Gl_vec(i);
        }
        Eigen::Tensor<double, 1> Gtau_cpp = cpp_sampling.evaluate(cpp_Gl);

        // Set up parameters for evaluation
        int ndim = 1;
        int dims[1] = {basis_size};
        int target_dim = 0;

        // Allocate memory for coefficients
        double* coeffs = (double*)malloc(basis_size * sizeof(double));
        // Create coefficients (simple test values)
        for (int i = 0; i < basis_size; i++) {
            coeffs[i] = cpp_Gl_vec(i);
        }

        // Create output buffer
        double* output = (double*)malloc(basis_size * sizeof(double));

        // Evaluate using C API
        int status = spir_sampling_evaluate_dd(
            sampling,
            SPIR_ORDER_ROW_MAJOR,  // Assuming this enum is defined in the header
            ndim,
            dims,
            target_dim,
            coeffs,
            output
        );

        for (int i = 0; i < basis_size; i++) {
            REQUIRE(output[i] == Approx(Gtau_cpp(i)));
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        // Free allocated memory
        free(coeffs);
        free(output);
    }
}
