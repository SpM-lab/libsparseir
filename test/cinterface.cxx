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

TEST_CASE("Kernel Accuracy Tests", "[kernel]")
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

TEST_CASE("Sampling", "[sampling]")
{
    SECTION("TauSampling Constructor")
    {
        double beta = 1.0;
        double wmax = 10.0;

        auto basis = spir_fermionic_basis_new(beta, wmax, 1e-15);
        REQUIRE(basis != nullptr);

        auto sampling = spir_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);
        
    }
}