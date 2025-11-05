#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <sparseir/sparseir.h>   // C interface

using Catch::Approx;

TEST_CASE("Test spir_kernel_get_sve_hints_nsvals", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    int nsvals;
    status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, &nsvals);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(nsvals > 0);
    
    // For lambda=10, log10(10)=1, so nsvals should be around (25+1)*1 = 26
    // But actual formula may vary, so just check it's reasonable
    REQUIRE(nsvals >= 10);
    REQUIRE(nsvals <= 1000);
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints_ngauss", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon_coarse = 1e-6;
    double epsilon_fine = 1e-10;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    int ngauss_coarse;
    status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon_coarse, &ngauss_coarse);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(ngauss_coarse > 0);
    
    int ngauss_fine;
    status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon_fine, &ngauss_fine);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(ngauss_fine > 0);
    
    // For epsilon >= 1e-8, ngauss should be 10
    // For epsilon < 1e-8, ngauss should be 16
    REQUIRE(ngauss_coarse == 10);
    REQUIRE(ngauss_fine == 16);
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints_segments_x", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    // First call: get the number of segments
    int n_segments = 0;
    status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, nullptr, &n_segments);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments > 0);
    
    // Second call: get the actual segments
    std::vector<double> segments(n_segments);
    int n_segments_out = n_segments;
    status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, segments.data(), &n_segments_out);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments_out == n_segments);
    
    // Verify segments are valid
    REQUIRE(segments.size() == static_cast<size_t>(n_segments));
    REQUIRE(segments[0] == Approx(-1.0).margin(1e-10));
    REQUIRE(segments[n_segments - 1] == Approx(1.0).margin(1e-10));
    
    // Verify segments are in ascending order
    for (int i = 1; i < n_segments; ++i) {
        REQUIRE(segments[i] > segments[i-1]);
    }
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints_segments_y", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    // First call: get the number of segments
    int n_segments = 0;
    status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, nullptr, &n_segments);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments > 0);
    
    // Second call: get the actual segments
    std::vector<double> segments(n_segments);
    int n_segments_out = n_segments;
    status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, segments.data(), &n_segments_out);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments_out == n_segments);
    
    // Verify segments are valid
    REQUIRE(segments.size() == static_cast<size_t>(n_segments));
    REQUIRE(segments[0] == Approx(-1.0).margin(1e-10));
    REQUIRE(segments[n_segments - 1] == Approx(1.0).margin(1e-10));
    
    // Verify segments are in ascending order
    for (int i = 1; i < n_segments; ++i) {
        REQUIRE(segments[i] > segments[i-1]);
    }
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints with RegularizedBoseKernel", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_reg_bose_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    // Test nsvals
    int nsvals;
    status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, &nsvals);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(nsvals > 0);
    
    // Test ngauss
    int ngauss;
    status = spir_kernel_get_sve_hints_ngauss(kernel, epsilon, &ngauss);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(ngauss > 0);
    
    // Test segments_x
    int n_segments_x = 0;
    status = spir_kernel_get_sve_hints_segments_x(kernel, epsilon, nullptr, &n_segments_x);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments_x > 0);
    
    // Test segments_y
    int n_segments_y = 0;
    status = spir_kernel_get_sve_hints_segments_y(kernel, epsilon, nullptr, &n_segments_y);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_segments_y > 0);
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_kernel_get_sve_hints error handling", "[cinterface]")
{
    double lambda = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(kernel != nullptr);
    
    // Test with nullptr kernel
    int nsvals;
    status = spir_kernel_get_sve_hints_nsvals(nullptr, epsilon, &nsvals);
    REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
    
    // Test with nullptr output parameter
    status = spir_kernel_get_sve_hints_nsvals(kernel, epsilon, nullptr);
    REQUIRE(status != SPIR_COMPUTATION_SUCCESS);
    
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_choose_working_type", "[cinterface]")
{
    // Test with epsilon >= 1e-8 -> should return FLOAT64
    {
        int twork = spir_choose_working_type(1e-6);
        REQUIRE(twork == SPIR_TWORK_FLOAT64);
    }
    
    {
        int twork = spir_choose_working_type(1e-8);
        REQUIRE(twork == SPIR_TWORK_FLOAT64);
    }
    
    // Test with epsilon < 1e-8 -> should return FLOAT64X2
    {
        int twork = spir_choose_working_type(1e-10);
        REQUIRE(twork == SPIR_TWORK_FLOAT64X2);
    }
    
    {
        int twork = spir_choose_working_type(1e-15);
        REQUIRE(twork == SPIR_TWORK_FLOAT64X2);
    }
    
    // Test with NaN -> should return FLOAT64X2
    {
        int twork = spir_choose_working_type(std::numeric_limits<double>::quiet_NaN());
        REQUIRE(twork == SPIR_TWORK_FLOAT64X2);
    }
    
    // Test boundary case: epsilon = 1e-8 exactly
    {
        int twork = spir_choose_working_type(1e-8);
        REQUIRE(twork == SPIR_TWORK_FLOAT64);
    }
    
    // Test boundary case: epsilon just below 1e-8
    {
        int twork = spir_choose_working_type(0.99e-8);
        REQUIRE(twork == SPIR_TWORK_FLOAT64X2);
    }
}
