#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <sparseir/sparseir.h>   // C interface
#include "xprec/ddouble-header-only.hpp"

using Catch::Approx;

TEST_CASE("Test spir_basis_get_default_matsus_ext", "[cinterface]")
{
    double beta = 10.0;
    double wmax = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(beta * wmax, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    spir_sve_result* sve = spir_sve_result_new(kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    spir_basis* basis = spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, wmax, epsilon, kernel, sve, -1, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    int basis_size;
    status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Test without mitigation
    {
        bool positive_only = false;
        bool mitigate = false;
        int n_points_requested = basis_size;
        
        int n_points_returned = 0;
        Eigen::Vector<int64_t, Eigen::Dynamic> points(n_points_requested + 10);
        
        status = spir_basis_get_default_matsus_ext(
            basis, positive_only, mitigate, n_points_requested, points.data(), &n_points_returned);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points_returned >= n_points_requested);
        REQUIRE(n_points_returned <= n_points_requested + 10);
        
        // Verify points are valid fermionic frequencies (odd integers)
        for (int i = 0; i < n_points_returned; ++i) {
            REQUIRE(llabs(points(i)) % 2 == 1);
        }
    }
    
    // Test with mitigation
    {
        bool positive_only = false;
        bool mitigate = true;
        int n_points_requested = basis_size;
        
        int n_points_returned = 0;
        Eigen::Vector<int64_t, Eigen::Dynamic> points(n_points_requested + 10);
        
        status = spir_basis_get_default_matsus_ext(
            basis, positive_only, mitigate, n_points_requested, points.data(), &n_points_returned);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points_returned >= n_points_requested);  // May exceed when mitigate is true
        
        // Verify points are valid fermionic frequencies
        for (int i = 0; i < n_points_returned; ++i) {
            REQUIRE(llabs(points(i)) % 2 == 1);
        }
    }
    
    // Test positive_only = true
    {
        bool positive_only = true;
        bool mitigate = false;
        // When positive_only=true, L represents total frequencies, so we request basis_size
        int n_points_requested = basis_size;
        
        int n_points_returned = 0;
        Eigen::Vector<int64_t, Eigen::Dynamic> points(n_points_requested + 10);
        
        status = spir_basis_get_default_matsus_ext(
            basis, positive_only, mitigate, n_points_requested, points.data(), &n_points_returned);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        // When positive_only=true, returned frequencies should be approximately n_points_requested / 2
        REQUIRE(n_points_returned > 0);
        REQUIRE(n_points_returned <= basis_size);
        
        // Verify all points are positive and odd
        for (int i = 0; i < n_points_returned; ++i) {
            REQUIRE(points(i) > 0);
            REQUIRE(points(i) % 2 == 1);
        }
    }
    
    spir_basis_release(basis);
    spir_sve_result_release(sve);
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_uhat_get_default_matsus", "[cinterface]")
{
    double beta = 10.0;
    double wmax = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(beta * wmax, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    spir_sve_result* sve = spir_sve_result_new(kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    spir_basis* basis = spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, wmax, epsilon, kernel, sve, -1, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    int basis_size;
    status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Get uhat from basis
    spir_funcs* uhat = spir_basis_get_uhat(basis, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(uhat != nullptr);
    
    // Test without mitigation
    {
        int L = basis_size;
        bool positive_only = false;
        bool mitigate = false;
        
        int n_points_returned = 0;
        Eigen::Vector<int64_t, Eigen::Dynamic> points(L + 10);
        
        status = spir_uhat_get_default_matsus(
            uhat, L, positive_only, mitigate, points.data(), &n_points_returned);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points_returned >= L);
        REQUIRE(n_points_returned <= L + 10);
        
        // Verify points are valid fermionic frequencies
        for (int i = 0; i < n_points_returned; ++i) {
            REQUIRE(llabs(points(i)) % 2 == 1);
        }
    }
    
    // Test with mitigation
    {
        int L = basis_size;
        bool positive_only = false;
        bool mitigate = true;
        
        int n_points_returned = 0;
        Eigen::Vector<int64_t, Eigen::Dynamic> points(L + 10);
        
        status = spir_uhat_get_default_matsus(
            uhat, L, positive_only, mitigate, points.data(), &n_points_returned);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points_returned >= L);  // May exceed when mitigate is true
        
        // Verify points are valid fermionic frequencies
        for (int i = 0; i < n_points_returned; ++i) {
            REQUIRE(llabs(points(i)) % 2 == 1);
        }
    }
    
    // Test bosonic statistics
    {
        spir_basis* basis_bosonic = spir_basis_new(SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon, kernel, sve, -1, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        spir_funcs* uhat_bosonic = spir_basis_get_uhat(basis_bosonic, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        int basis_size_bosonic;
        status = spir_basis_get_size(basis_bosonic, &basis_size_bosonic);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        int L = basis_size_bosonic;
        bool positive_only = false;
        bool mitigate = false;
        
        int n_points_returned = 0;
        Eigen::Vector<int64_t, Eigen::Dynamic> points(L + 10);
        
        status = spir_uhat_get_default_matsus(
            uhat_bosonic, L, positive_only, mitigate, points.data(), &n_points_returned);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points_returned >= L);
        
        // Verify points are valid bosonic frequencies (even integers, including 0)
        for (int i = 0; i < n_points_returned; ++i) {
            REQUIRE(llabs(points(i)) % 2 == 0);
        }
        
        spir_funcs_release(uhat_bosonic);
        spir_basis_release(basis_bosonic);
    }
    
    spir_funcs_release(uhat);
    spir_basis_release(basis);
    spir_sve_result_release(sve);
    spir_kernel_release(kernel);
}

TEST_CASE("Test spir_basis_get_uhat_full", "[cinterface]")
{
    double beta = 10.0;
    double wmax = 10.0;
    double epsilon = 1e-8;
    
    int status;
    spir_kernel* kernel = spir_logistic_kernel_new(beta * wmax, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    spir_sve_result* sve = spir_sve_result_new(kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    spir_basis* basis = spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, wmax, epsilon, kernel, sve, -1, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    int basis_size;
    status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Get uhat and uhat_full from basis
    spir_funcs* uhat = spir_basis_get_uhat(basis, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(uhat != nullptr);
    
    spir_funcs* uhat_full = spir_basis_get_uhat_full(basis, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(uhat_full != nullptr);
    
    // Get sizes
    int uhat_size;
    status = spir_funcs_get_size(uhat, &uhat_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    int uhat_full_size;
    status = spir_funcs_get_size(uhat_full, &uhat_full_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Verify uhat_full size >= uhat size
    REQUIRE(uhat_full_size >= uhat_size);
    REQUIRE(uhat_size == basis_size);
    
    // Test bosonic statistics
    {
        spir_basis* basis_bosonic = spir_basis_new(SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon, kernel, sve, -1, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        spir_funcs* uhat_bosonic = spir_basis_get_uhat(basis_bosonic, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        spir_funcs* uhat_full_bosonic = spir_basis_get_uhat_full(basis_bosonic, &status);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(uhat_full_bosonic != nullptr);
        
        int uhat_bosonic_size;
        status = spir_funcs_get_size(uhat_bosonic, &uhat_bosonic_size);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        int uhat_full_bosonic_size;
        status = spir_funcs_get_size(uhat_full_bosonic, &uhat_full_bosonic_size);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        REQUIRE(uhat_full_bosonic_size >= uhat_bosonic_size);
        
        spir_funcs_release(uhat_full_bosonic);
        spir_funcs_release(uhat_bosonic);
        spir_basis_release(basis_bosonic);
    }
    
    spir_funcs_release(uhat_full);
    spir_funcs_release(uhat);
    spir_basis_release(basis);
    spir_sve_result_release(sve);
    spir_kernel_release(kernel);
}

// Helper functions for testing FunctionKernel
// These wrap LogisticKernel functionality for testing
// Note: In practice, Julia side will use closures/anonymous functions

// For testing, we need to capture lambda from a global or closure
// In practice, Julia side will use closures/anonymous functions
static double test_lambda_dd = 100.0;

// Template function for kernel computation using LogisticKernel's compute_from_uv logic
template <typename T>
T test_kernel_compute_impl(T x, T y, double lambda)
{
    using std::abs;
    using std::exp;
    
    // Use LogisticKernel's compute_from_uv logic for numerical stability
    // Domain is [-1.0, 1.0] for both x and y
    T xmin = T(-1.0);
    T xmax = T(1.0);
    
    // Compute x_plus and x_minus (used for numerical stability)
    T x_plus = x - xmin;  // = x - (-1.0) = x + 1.0
    T x_minus = xmax - x;  // = 1.0 - x
    
    // Compute u_plus, u_minus, v (following LogisticKernel::compute_uv)
    T u_plus = T(0.5) * x_plus;
    T u_minus = T(0.5) * x_minus;
    T v = T(lambda) * y;
    
    // Compute kernel value using compute_from_uv logic
    T mabs_v = -abs(v);
    T numerator;
    if (static_cast<double>(v) >= 0) {
        numerator = exp(u_plus * mabs_v);
    } else {
        numerator = exp(u_minus * mabs_v);
    }
    T denominator = T(1.0) + exp(mabs_v);
    
    return numerator / denominator;
}

// Specialization for xprec::DDouble
template <>
xprec::DDouble test_kernel_compute_impl<xprec::DDouble>(xprec::DDouble x, xprec::DDouble y, double lambda)
{
    // Use LogisticKernel's compute_from_uv logic for numerical stability
    // Domain is [-1.0, 1.0] for both x and y
    xprec::DDouble xmin(-1.0);
    xprec::DDouble xmax(1.0);
    
    // Compute x_plus and x_minus (used for numerical stability)
    xprec::DDouble x_plus = x - xmin;  // = x - (-1.0) = x + 1.0
    xprec::DDouble x_minus = xmax - x;  // = 1.0 - x
    
    // Compute u_plus, u_minus, v (following LogisticKernel::compute_uv)
    xprec::DDouble u_plus = 0.5 * x_plus;
    xprec::DDouble u_minus = 0.5 * x_minus;
    xprec::DDouble v = xprec::DDouble(lambda) * y;
    
    // Compute kernel value using compute_from_uv logic
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

// Batch function wrappers for C-API
static void test_batch_func(const double* xs, const double* ys, int n,
                           double* out, void* user_data)
{
    (void)user_data;  // Not used in test
    for (int i = 0; i < n; ++i) {
        out[i] = test_kernel_compute_impl<double>(xs[i], ys[i], test_lambda_dd);
    }
}

static void test_batch_func_dd(const double* xs_hi, const double* xs_lo,
                               const double* ys_hi, const double* ys_lo,
                               int n,
                               double* out_hi, double* out_lo,
                               void* user_data)
{
    (void)user_data;  // Not used in test
    for (int i = 0; i < n; ++i) {
        xprec::DDouble x(xs_hi[i], xs_lo[i]);
        xprec::DDouble y(ys_hi[i], ys_lo[i]);
        xprec::DDouble result = test_kernel_compute_impl<xprec::DDouble>(x, y, test_lambda_dd);
        out_hi[i] = result.hi();
        out_lo[i] = result.lo();
    }
}

static void test_segments_x_func(double epsilon, double* segments, int* n_segments, void* user_data)
{
    double lambda = test_lambda_dd;
    
    // Use LogisticKernel's segments_x logic
    int nzeros = std::max(static_cast<int>(std::round(15 * std::log10(lambda))), 1);
    
    if (segments == nullptr) {
        // First call: return the number of segments
        *n_segments = 2 * nzeros + 1;
    } else {
        // Second call: fill the segments
        std::vector<double> temp(nzeros);
        for (int i = 0; i < nzeros; ++i) {
            temp[i] = 0.143 * i;
        }
        
        std::vector<double> diffs(nzeros);
        for (int i = 0; i < nzeros; ++i) {
            diffs[i] = 1.0 / std::cosh(temp[i]);
        }
        
        std::vector<double> zeros(nzeros);
        zeros[0] = diffs[0];
        for (int i = 1; i < nzeros; ++i) {
            zeros[i] = zeros[i - 1] + diffs[i];
        }
        
        double last_zero = zeros.back();
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] /= last_zero;
        }
        
        // Create symmetric segments
        for (int i = 0; i < nzeros; ++i) {
            segments[i] = -zeros[nzeros - i - 1];
            segments[nzeros + i + 1] = zeros[i];
        }
        segments[nzeros] = 0.0;
    }
}

static void test_segments_y_func(double epsilon, double* segments, int* n_segments, void* user_data)
{
    double lambda = test_lambda_dd;
    
    // Use LogisticKernel's segments_y logic
    int nzeros = std::max(static_cast<int>(std::round(15 * std::log10(lambda))), 1);
    
    if (segments == nullptr) {
        // First call: return the number of segments
        *n_segments = 2 * nzeros + 2;
    } else {
        // Second call: fill the segments
        std::vector<double> temp(nzeros);
        for (int i = 0; i < nzeros; ++i) {
            temp[i] = 0.143 * i;
        }
        
        std::vector<double> diffs(nzeros);
        for (int i = 0; i < nzeros; ++i) {
            diffs[i] = 1.0 / std::cosh(temp[i]);
        }
        
        std::vector<double> zeros(nzeros);
        zeros[0] = diffs[0];
        for (int i = 1; i < nzeros; ++i) {
            zeros[i] = zeros[i - 1] + diffs[i];
        }
        
        double last_zero = zeros.back();
        for (int i = 0; i < nzeros; ++i) {
            zeros[i] /= last_zero;
        }
        
        // Create symmetric segments with endpoints
        segments[0] = -1.0;
        for (int i = 0; i < nzeros; ++i) {
            segments[i + 1] = -zeros[nzeros - i - 1];
            segments[nzeros + i + 2] = zeros[i];
        }
        segments[nzeros + 1] = 0.0;
        segments[2 * nzeros + 1] = 1.0;
    }
}

static int test_nsvals_func(double epsilon, void* user_data)
{
    (void)user_data;  // Not used in test
    double lambda = test_lambda_dd;
    double log10_Lambda = std::max(1.0, std::log10(lambda));
    return static_cast<int>(std::round((25 + log10_Lambda) * log10_Lambda));
}

static int test_ngauss_func(double epsilon, void* user_data)
{
    (void)user_data;  // Not used in test
    return epsilon >= 1e-8 ? 10 : 16;
}

static double test_weight_func_fermionic(double beta, double omega, void* user_data)
{
    (void)user_data;  // Not used in test
    (void)beta;
    (void)omega;
    // LogisticKernel's fermionic weight function is 1.0
    return 1.0;
}

static double test_weight_func_bosonic(double beta, double omega, void* user_data)
{
    (void)user_data;  // Not used in test
    // LogisticKernel's bosonic weight function is 1.0 / tanh(Λ y / 2)
    // where y = beta * omega / lambda, so Λ y / 2 = lambda * beta * omega / (2 * lambda) = beta * omega / 2
    double lambda = test_lambda_dd;
    double v_half = lambda * 0.5 * (beta * omega / lambda);  // y = beta * omega / lambda
    if (std::abs(v_half) < 1e-10) {
        return 1.0;  // Avoid division by zero
    }
    return 1.0 / std::tanh(v_half);
}

TEST_CASE("Test spir_function_kernel_new", "[cinterface]")
{
    double lambda = 100.0;
    
    // Set lambda for functions (in practice, Julia will use closures)
    test_lambda_dd = lambda;
    
    // Test creating a function kernel
    int status;
    spir_kernel* func_kernel = spir_function_kernel_new(
        lambda,
        test_batch_func,
        test_batch_func_dd,  // Extended precision batch function pointer
        -1.0, 1.0,  // xmin, xmax
        -1.0, 1.0,  // ymin, ymax
        1,  // is_centrosymmetric = true
        test_segments_x_func,
        test_segments_y_func,
        test_nsvals_func,
        test_ngauss_func,
        test_weight_func_fermionic,
        test_weight_func_bosonic,
        nullptr,  // user_data (not used in test)
        &status);
    
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(func_kernel != nullptr);
    
    // Test kernel domain
    double xmin, xmax, ymin, ymax;
    int domain_status = spir_kernel_domain(func_kernel, &xmin, &xmax, &ymin, &ymax);
    REQUIRE(domain_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(xmin == -1.0);
    REQUIRE(xmax == 1.0);
    REQUIRE(ymin == -1.0);
    REQUIRE(ymax == 1.0);
    
    // Compare with LogisticKernel
    int logistic_status;
    spir_kernel* logistic_kernel = spir_logistic_kernel_new(lambda, &logistic_status);
    REQUIRE(logistic_status == SPIR_COMPUTATION_SUCCESS);
    
    // Test SVE computation with function kernel
    double epsilon = 1e-8;
    int sve_status;
    spir_sve_result* sve_func = spir_sve_result_new(
        func_kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &sve_status);
    REQUIRE(sve_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sve_func != nullptr);
    
    // Compare SVE result size with LogisticKernel
    spir_sve_result* sve_logistic = spir_sve_result_new(
        logistic_kernel, epsilon, -1, -1, SPIR_TWORK_AUTO, &sve_status);
    REQUIRE(sve_status == SPIR_COMPUTATION_SUCCESS);
    
    int size_func = 0;
    int size_status_func = spir_sve_result_get_size(sve_func, &size_func);
    REQUIRE(size_status_func == SPIR_COMPUTATION_SUCCESS);
    
    int size_logistic = 0;
    int size_status_logistic = spir_sve_result_get_size(sve_logistic, &size_logistic);
    REQUIRE(size_status_logistic == SPIR_COMPUTATION_SUCCESS);
    
    // Sizes should match exactly since both kernels use the same SVE hints
    REQUIRE(size_func == size_logistic);
    
    // Test creating basis with function kernel
    double beta = 10.0;
    double wmax = 10.0;
    spir_basis* basis_func = spir_basis_new(
        SPIR_STATISTICS_FERMIONIC, beta, wmax, epsilon,
        func_kernel, sve_func, -1, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis_func != nullptr);
    
    // Create basis with LogisticKernel for comparison
    spir_basis* basis_logistic = spir_basis_new(
        SPIR_STATISTICS_FERMIONIC, beta, wmax, epsilon,
        logistic_kernel, sve_logistic, -1, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis_logistic != nullptr);
    
    int basis_size_func, basis_size_logistic;
    status = spir_basis_get_size(basis_func, &basis_size_func);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    status = spir_basis_get_size(basis_logistic, &basis_size_logistic);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    
    // Basis sizes should match
    REQUIRE(basis_size_func == basis_size_logistic);
    int basis_size = basis_size_func;
    
    // Get basis functions for comparison
    spir_funcs* u_func = spir_basis_get_u(basis_func, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(u_func != nullptr);
    spir_funcs* u_logistic = spir_basis_get_u(basis_logistic, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(u_logistic != nullptr);
    
    // Compare kernel values by evaluating basis functions at various points
    // Since the kernel values are used in SVE, comparing basis functions should
    // verify kernel consistency
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> tau_dist(0.0, beta);
    
    const int n_test_points = 50;
    for (int i = 0; i < n_test_points; ++i) {
        double tau = tau_dist(gen);
        
        // Evaluate basis functions at this point
        std::vector<double> u_vals_func(basis_size);
        std::vector<double> u_vals_logistic(basis_size);
        
        status = spir_funcs_eval(u_func, tau, u_vals_func.data());
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        status = spir_funcs_eval(u_logistic, tau, u_vals_logistic.data());
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        
        // Compare basis function values (they should match if kernels are equivalent)
        // Note: Small numerical differences are expected due to different computation paths.
        // The error margin of 1e-6 may be too loose for epsilon=1e-8.
        for (int j = 0; j < basis_size; ++j) {
            REQUIRE(u_vals_func[j] == Approx(u_vals_logistic[j]).margin(1e-6));
        }
    }
    
    // Clean up
    spir_funcs_release(u_func);
    spir_funcs_release(u_logistic);
    spir_basis_release(basis_func);
    spir_basis_release(basis_logistic);
    spir_sve_result_release(sve_func);
    spir_sve_result_release(sve_logistic);
    spir_kernel_release(func_kernel);
    spir_kernel_release(logistic_kernel);
}

