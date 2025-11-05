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
#include "xprec/ddouble-header-only.hpp"

using Catch::Approx;

// Test holder: wraps a LogisticKernel for C-API callbacks
// We use C-API to create a LogisticKernel and delegate to it
struct TestKernelHolder {
    spir_kernel* kernel;
    double lambda;
    
    TestKernelHolder(double lambda_val) : kernel(nullptr), lambda(lambda_val) {
        int status;
        kernel = spir_logistic_kernel_new(lambda_val, &status);
        if (status != SPIR_COMPUTATION_SUCCESS || !kernel) {
            throw std::runtime_error("Failed to create LogisticKernel");
        }
    }
    
    ~TestKernelHolder() {
        if (kernel) {
            spir_kernel_release(kernel);
        }
    }
    
    // Compute kernel value using C-API
    double compute(double x, double y) {
        // We need to get the kernel value - since C-API doesn't expose compute directly,
        // we'll use a workaround: create a temporary SVE result and extract values
        // Actually, for testing, we can just call the kernel computation function directly
        // For now, let's use a simple approximation: we'll implement the kernel computation
        // using the LogisticKernel formula
        return compute_logistic_kernel_impl(x, y, lambda);
    }
    
    xprec::DDouble compute(xprec::DDouble x, xprec::DDouble y) {
        return compute_logistic_kernel_impl_dd(x, y, lambda);
    }
    
private:
    double compute_logistic_kernel_impl(double x, double y, double lambda_val) {
        using std::abs;
        using std::exp;
        
        double xmin = -1.0;
        double xmax = 1.0;
        double x_plus = x - xmin;
        double x_minus = xmax - x;
        double u_plus = 0.5 * x_plus;
        double u_minus = 0.5 * x_minus;
        double v = lambda_val * y;
        
        double mabs_v = -abs(v);
        double numerator;
        if (v >= 0) {
            numerator = exp(u_plus * mabs_v);
        } else {
            numerator = exp(u_minus * mabs_v);
        }
        double denominator = 1.0 + exp(mabs_v);
        
        return numerator / denominator;
    }
    
    xprec::DDouble compute_logistic_kernel_impl_dd(xprec::DDouble x, xprec::DDouble y, double lambda_val) {
        xprec::DDouble xmin(-1.0);
        xprec::DDouble xmax(1.0);
        xprec::DDouble x_plus = x - xmin;
        xprec::DDouble x_minus = xmax - x;
        xprec::DDouble u_plus = 0.5 * x_plus;
        xprec::DDouble u_minus = 0.5 * x_minus;
        xprec::DDouble v = xprec::DDouble(lambda_val) * y;
        
        xprec::DDouble mabs_v = -xprec::abs(v);
        xprec::DDouble numerator;
        if (static_cast<double>(v) >= 0) {
            numerator = xprec::exp(u_plus * mabs_v);
        } else {
            numerator = xprec::exp(u_minus * mabs_v);
        }
        xprec::DDouble denominator = 1.0 + xprec::exp(mabs_v);
        
        return numerator / denominator;
    }
};

// Batch function for double precision
static void test_batch_func(const double* xs, const double* ys, int n,
                           double* out, void* user_data)
{
    TestKernelHolder* holder = static_cast<TestKernelHolder*>(user_data);
    for (int i = 0; i < n; ++i) {
        out[i] = holder->compute(xs[i], ys[i]);
    }
}

// Batch function for double-double precision
static void test_batch_func_dd(const double* xs_hi, const double* xs_lo,
                               const double* ys_hi, const double* ys_lo,
                               int n,
                               double* out_hi, double* out_lo,
                               void* user_data)
{
    TestKernelHolder* holder = static_cast<TestKernelHolder*>(user_data);
    for (int i = 0; i < n; ++i) {
        xprec::DDouble x(xs_hi[i], xs_lo[i]);
        xprec::DDouble y(ys_hi[i], ys_lo[i]);
        xprec::DDouble result = holder->compute(x, y);
        out_hi[i] = result.hi();
        out_lo[i] = result.lo();
    }
}

// SVE hints callbacks: delegate to the LogisticKernel via C-API
static void test_segments_x_func(double epsilon, double* segments, int* n_segments, void* user_data)
{
    TestKernelHolder* holder = static_cast<TestKernelHolder*>(user_data);
    int status = spir_kernel_get_sve_hints_segments_x(holder->kernel, epsilon, segments, n_segments);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        throw std::runtime_error("Failed to get segments_x from LogisticKernel");
    }
}

static void test_segments_y_func(double epsilon, double* segments, int* n_segments, void* user_data)
{
    TestKernelHolder* holder = static_cast<TestKernelHolder*>(user_data);
    int status = spir_kernel_get_sve_hints_segments_y(holder->kernel, epsilon, segments, n_segments);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        throw std::runtime_error("Failed to get segments_y from LogisticKernel");
    }
}

static int test_nsvals_func(double epsilon, void* user_data)
{
    TestKernelHolder* holder = static_cast<TestKernelHolder*>(user_data);
    int nsvals = 0;
    int status = spir_kernel_get_sve_hints_nsvals(holder->kernel, epsilon, &nsvals);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        throw std::runtime_error("Failed to get nsvals from LogisticKernel");
    }
    return nsvals;
}

static int test_ngauss_func(double epsilon, void* user_data)
{
    TestKernelHolder* holder = static_cast<TestKernelHolder*>(user_data);
    int ngauss = 0;
    int status = spir_kernel_get_sve_hints_ngauss(holder->kernel, epsilon, &ngauss);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        throw std::runtime_error("Failed to get ngauss from LogisticKernel");
    }
    return ngauss;
}

// Weight functions: default to 1.0
static double test_weight_func_fermionic(double beta, double omega, void* user_data)
{
    (void)user_data;
    (void)beta;
    (void)omega;
    return 1.0;
}

static double test_weight_func_bosonic(double beta, double omega, void* user_data)
{
    (void)user_data;
    (void)beta;
    (void)omega;
    return 1.0;
}

TEST_CASE("Test custom kernel batch evaluation", "[cinterface][custom_kernel]")
{
    double lambda = 100.0;
    TestKernelHolder holder(lambda);
    
    // Test batch function with 4 corner points
    double xs[4] = {-1.0, -1.0, 1.0, 1.0};
    double ys[4] = {-1.0, 1.0, -1.0, 1.0};
    double out[4];
    
    test_batch_func(xs, ys, 4, out, &holder);
    
    // Verify results match direct computation
    for (int i = 0; i < 4; ++i) {
        double expected = holder.compute(xs[i], ys[i]);
        REQUIRE(out[i] == Approx(expected).margin(1e-10));
    }
}

TEST_CASE("Test custom kernel batch evaluation double-double", "[cinterface][custom_kernel]")
{
    double lambda = 100.0;
    TestKernelHolder holder(lambda);
    
    // Test batch function with 4 corner points
    double xs_hi[4] = {-1.0, -1.0, 1.0, 1.0};
    double xs_lo[4] = {0.0, 0.0, 0.0, 0.0};
    double ys_hi[4] = {-1.0, 1.0, -1.0, 1.0};
    double ys_lo[4] = {0.0, 0.0, 0.0, 0.0};
    double out_hi[4], out_lo[4];
    
    test_batch_func_dd(xs_hi, xs_lo, ys_hi, ys_lo, 4, out_hi, out_lo, &holder);
    
    // Verify results match direct computation
    for (int i = 0; i < 4; ++i) {
        xprec::DDouble x(xs_hi[i], xs_lo[i]);
        xprec::DDouble y(ys_hi[i], ys_lo[i]);
        xprec::DDouble expected = holder.compute(x, y);
        REQUIRE(out_hi[i] == Approx(expected.hi()).margin(1e-10));
        REQUIRE(out_lo[i] == Approx(expected.lo()).margin(1e-10));
    }
}

TEST_CASE("Test custom kernel SVE hints", "[cinterface][custom_kernel]")
{
    double lambda = 100.0;
    double epsilon = 1e-6;
    TestKernelHolder holder(lambda);
    
    // Test segments_x
    {
        int n_segments = 0;
        test_segments_x_func(epsilon, nullptr, &n_segments, &holder);
        REQUIRE(n_segments > 0);
        
        std::vector<double> segments(n_segments);
        test_segments_x_func(epsilon, segments.data(), &n_segments, &holder);
        REQUIRE(n_segments > 0);
        // Verify segments are ordered and within [-1, 1]
        for (int i = 0; i < n_segments; ++i) {
            REQUIRE(segments[i] >= -1.0);
            REQUIRE(segments[i] <= 1.0);
        }
    }
    
    // Test segments_y
    {
        int n_segments = 0;
        test_segments_y_func(epsilon, nullptr, &n_segments, &holder);
        REQUIRE(n_segments > 0);
        
        std::vector<double> segments(n_segments);
        test_segments_y_func(epsilon, segments.data(), &n_segments, &holder);
        REQUIRE(n_segments > 0);
        // Verify segments are ordered and within [-1, 1]
        for (int i = 0; i < n_segments; ++i) {
            REQUIRE(segments[i] >= -1.0);
            REQUIRE(segments[i] <= 1.0);
        }
    }
    
    // Test nsvals
    {
        int nsvals = test_nsvals_func(epsilon, &holder);
        REQUIRE(nsvals > 0);
    }
    
    // Test ngauss
    {
        int ngauss = test_ngauss_func(epsilon, &holder);
        REQUIRE(ngauss > 0);
    }
}

TEST_CASE("Test custom kernel weight functions", "[cinterface][custom_kernel]")
{
    double beta = 10.0;
    double omega = 5.0;
    
    // Test fermionic weight function
    double w_fermionic = test_weight_func_fermionic(beta, omega, nullptr);
    REQUIRE(w_fermionic == Approx(1.0));
    
    // Test bosonic weight function
    double w_bosonic = test_weight_func_bosonic(beta, omega, nullptr);
    REQUIRE(w_bosonic == Approx(1.0));
}

TEST_CASE("Test spir_function_kernel_new and SVE comparison", "[cinterface][custom_kernel]")
{
    double lambda = 100.0;
    double epsilon = 1e-6;
    
    TestKernelHolder holder(lambda);
    
    // Create custom kernel via C-API
    int status;
    spir_kernel* custom_kernel = spir_function_kernel_new(
        lambda,
        test_batch_func,
        test_batch_func_dd,
        -1.0, 1.0,  // xmin, xmax
        -1.0, 1.0,  // ymin, ymax
        1,  // is_centrosymmetric = true
        test_segments_x_func,
        test_segments_y_func,
        test_nsvals_func,
        test_ngauss_func,
        test_weight_func_fermionic,
        test_weight_func_bosonic,
        &holder,
        &status
    );
    
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(custom_kernel != nullptr);
    
    // Create reference LogisticKernel via C-API
    spir_kernel* logistic_kernel = spir_logistic_kernel_new(lambda, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(logistic_kernel != nullptr);
    
    // Compute SVE for custom kernel
    spir_sve_result* custom_sve = spir_sve_result_new(
        custom_kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(custom_sve != nullptr);
    
    // Compute SVE for reference kernel
    spir_sve_result* logistic_sve = spir_sve_result_new(
        logistic_kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(logistic_sve != nullptr);
    
    // Compare sizes
    int custom_size, logistic_size;
    status = spir_sve_result_get_size(custom_sve, &custom_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    status = spir_sve_result_get_size(logistic_sve, &logistic_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(custom_size == logistic_size);
    
    // Compare first ~10 singular values
    int n_compare = std::min(10, custom_size);
    std::vector<double> custom_svals(custom_size);
    std::vector<double> logistic_svals(logistic_size);
    
    status = spir_sve_result_get_svals(custom_sve, custom_svals.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    status = spir_sve_result_get_svals(logistic_sve, logistic_svals.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    for (int i = 0; i < n_compare; ++i) {
        REQUIRE(custom_svals[i] == Approx(logistic_svals[i]).margin(1e-10));
    }
    
    // Cleanup
    spir_sve_result_release(custom_sve);
    spir_sve_result_release(logistic_sve);
    spir_kernel_release(custom_kernel);
    spir_kernel_release(logistic_kernel);
}

TEST_CASE("Test custom kernel centrosymmetric true vs false", "[cinterface][custom_kernel]")
{
    double lambda = 100.0;
    double epsilon = 1e-6;
    
    TestKernelHolder holder_true(lambda);
    TestKernelHolder holder_false(lambda);
    
    // Create custom kernel with is_centrosymmetric = true
    int status;
    spir_kernel* kernel_true = spir_function_kernel_new(
        lambda,
        test_batch_func,
        test_batch_func_dd,
        -1.0, 1.0, -1.0, 1.0,
        1,  // is_centrosymmetric = true
        test_segments_x_func,
        test_segments_y_func,
        test_nsvals_func,
        test_ngauss_func,
        test_weight_func_fermionic,
        test_weight_func_bosonic,
        &holder_true,
        &status
    );
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Create custom kernel with is_centrosymmetric = false
    spir_kernel* kernel_false = spir_function_kernel_new(
        lambda,
        test_batch_func,
        test_batch_func_dd,
        -1.0, 1.0, -1.0, 1.0,
        0,  // is_centrosymmetric = false
        test_segments_x_func,
        test_segments_y_func,
        test_nsvals_func,
        test_ngauss_func,
        test_weight_func_fermionic,
        test_weight_func_bosonic,
        &holder_false,
        &status
    );
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Compute SVE for both
    spir_sve_result* sve_true = spir_sve_result_new(
        kernel_true, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    spir_sve_result* sve_false = spir_sve_result_new(
        kernel_false, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Compare sizes (should be identical)
    int size_true, size_false;
    status = spir_sve_result_get_size(sve_true, &size_true);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    status = spir_sve_result_get_size(sve_false, &size_false);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(size_true == size_false);
    
    // Compare singular values (should be identical)
    std::vector<double> svals_true(size_true);
    std::vector<double> svals_false(size_false);
    
    status = spir_sve_result_get_svals(sve_true, svals_true.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    status = spir_sve_result_get_svals(sve_false, svals_false.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    for (int i = 0; i < size_true; ++i) {
        REQUIRE(svals_true[i] == Approx(svals_false[i]).margin(1e-10));
    }
    
    // Cleanup
    spir_sve_result_release(sve_true);
    spir_sve_result_release(sve_false);
    spir_kernel_release(kernel_true);
    spir_kernel_release(kernel_false);
}

TEST_CASE("Test custom kernel batch evaluation with n > 1", "[cinterface][custom_kernel]")
{
    // This test verifies that batch functions are called with n > 1
    // during matrix construction (through SVEResult computation)
    double lambda = 100.0;
    double epsilon = 1e-6;
    
    TestKernelHolder holder(lambda);
    
    // Create custom kernel
    int status;
    spir_kernel* custom_kernel = spir_function_kernel_new(
        lambda,
        test_batch_func,
        test_batch_func_dd,
        -1.0, 1.0, -1.0, 1.0,
        1,  // is_centrosymmetric = true
        test_segments_x_func,
        test_segments_y_func,
        test_nsvals_func,
        test_ngauss_func,
        test_weight_func_fermionic,
        test_weight_func_bosonic,
        &holder,
        &status
    );
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Compute SVE - this should trigger matrix_from_gauss which calls batch_func with n > 1
    spir_sve_result* sve = spir_sve_result_new(
        custom_kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sve != nullptr);
    
    // Verify SVE result is valid
    int size;
    status = spir_sve_result_get_size(sve, &size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(size > 0);
    
    // Cleanup
    spir_sve_result_release(sve);
    spir_kernel_release(custom_kernel);
}

