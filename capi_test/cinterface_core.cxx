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

