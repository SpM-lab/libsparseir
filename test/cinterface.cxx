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
        spir_kernel* kernel = spir_kernel_logistic_new(9);
        REQUIRE(kernel != nullptr);

        // Get domain bounds
        double xmin, xmax, ymin, ymax;
        int status = spir_kernel_domain(kernel, &xmin, &xmax, &ymin, &ymax);
        REQUIRE(status == 0);

        // Compare with C++ implementation
        //auto cpp_kernel = sparseir::LogisticKernel(9);
        //auto cpp_xmin = cpp_kernel.xmin();
        //auto cpp_xmax = cpp_kernel.xmax();
        //auto cpp_ymin = cpp_kernel.ymin();
        ////auto cpp_ymax = cpp_kernel.ymax();
//
        //REQUIRE(xmin == Approx(cpp_xmin));
        //REQUIRE(xmax == Approx(cpp_xmax));
        //REQUIRE(ymin == Approx(cpp_ymin));
        //REQUIRE(ymax == Approx(cpp_ymax));

        // Clean up
        //spir_kernel_delete(kernel);
    }
}