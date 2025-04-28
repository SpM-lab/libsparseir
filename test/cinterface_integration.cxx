#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <sparseir/sparseir.hpp>   // C++ interface
#include <sparseir/sparseir.h>   // C interface

using Catch::Approx;

template <typename S>
spir_statistics_type get_stat()
{
    if (std::is_same<S, sparseir::Fermionic>::value) {
        return SPIR_STATISTICS_FERMIONIC;
    } else {
        return SPIR_STATISTICS_BOSONIC;
    }
}

template <typename S, typename K>
void integration_test(double beta, double wmax, double epsilon, int ndim, int dim)
{
    auto stat = get_stat<S>();
    int32_t status;

    // IR basis
    spir_finite_temp_basis *basis =
        spir_finite_temp_basis_new(stat, beta, wmax, epsilon);
    REQUIRE(basis != nullptr);
    int32_t basis_size;
    status = spir_finite_temp_basis_get_size(basis, &basis_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // DLR
    spir_dlr *dlr = spir_dlr_new(basis);
    REQUIRE(dlr != nullptr);
    int32_t npoles;
    status = spir_dlr_get_num_poles(dlr, &npoles);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(npoles >= 0);
    REQUIRE(npoles >= basis_size);
    Eigen::VectorXd poles(npoles);
    status = spir_dlr_get_poles(dlr, poles.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Generate random DLR coefficients
    Eigen::VectorXd coeffs(npoles); // Extend to Eigen::Tensor
    std::mt19937 gen(982743);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < npoles; i++) {
        // We scale the random coefficients by the square root of the absolute value of the poles
        // to ensure that all the poles contribute to G(tau) equally.
        coeffs(i) = (2.0 * dis(gen) - 1.0) * std::sqrt(std::abs(poles(i)));
    }
    REQUIRE(poles.array().abs().maxCoeff() <= wmax);

    // Convert DLR coefficients to IR coefficients
    Eigen::VectorXd g_IR(basis_size); // Extend to Eigen::Tensor
    status = spir_dlr_to_IR(dlr, SPIR_ORDER_COLUMN_MAJOR, 1, &basis_size, 0, coeffs.data(), g_IR.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
}