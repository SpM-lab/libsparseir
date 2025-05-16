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


template <typename S>
spir_statistics_type get_stat()
{
    if (std::is_same<S, sparseir::Fermionic>::value) {
        return SPIR_STATISTICS_FERMIONIC;
    } else {
        return SPIR_STATISTICS_BOSONIC;
    }
}


// Compression Tests
/*
template <typename Statistics>
void test_finite_temp_basis_dlr()
{
    const double beta = 10000.0;
    const auto stat = get_stat<Statistics>();
    const double wmax = 1.0;
    const double epsilon = 1e-12;

    int32_t status;

    int32_t basis_status;
    spir_basis *basis = spir_basis_new(stat, beta, wmax, epsilon, &basis_status);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);
    int32_t basis_size;
    basis_status = spir_basis_get_size(basis, &basis_size);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis_size >= 0);

    // IR sampling points
    int32_t n_smpl_points;
    status = spir_basis_get_num_default_matsubara_sampling_points(basis, false, &n_smpl_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_smpl_points >= 0);
    std::vector<int32_t> smpl_points(n_smpl_points);
    status = spir_basis_get_default_matsubara_sampling_points(basis, false, smpl_points.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);



    int32_t dlr_status;
    spir_basis *dlr = spir_dlr_new(basis, &dlr_status);
    REQUIRE(dlr_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(dlr != nullptr);

    int32_t num_poles;
    dlr_status = spir_dlr_get_num_poles(dlr, &num_poles);
    REQUIRE(dlr_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(num_poles >= 0);
    REQUIRE(num_poles >= basis_size);

    {
        double *poles = (double *)malloc(num_poles * sizeof(double));
        dlr_status = spir_dlr_get_poles(dlr, poles);
        REQUIRE(dlr_status == SPIR_COMPUTATION_SUCCESS);
        free(poles);
    }

    const int npoles = 10;
    Eigen::VectorXd poles(npoles);
    Eigen::VectorXd coeffs(npoles);
    std::mt19937 gen(982743);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < npoles; i++) {
        poles(i) = wmax * (2.0 * dis(gen) - 1.0);
        coeffs(i) = 2.0 * dis(gen) - 1.0;
    }
    REQUIRE(poles.array().abs().maxCoeff() <= wmax);

    int32_t poles_status;
    spir_basis *dlr_with_poles = spir_dlr_new_with_poles(basis, npoles, poles.data(), &poles_status);
    REQUIRE(poles_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(dlr_with_poles != nullptr);

    double *Gl = (double *)malloc(basis_size * sizeof(double));
    int32_t to_ir_input_dims[1] = {npoles};
    int32_t ndim = 1;
    int32_t target_dim = 0;
    int status_to_IR = spir_dlr_to_ir_dd(dlr_with_poles, SPIR_ORDER_COLUMN_MAJOR,
                                      ndim, to_ir_input_dims, target_dim, coeffs.data(), Gl);

    REQUIRE(status_to_IR == SPIR_COMPUTATION_SUCCESS);
    double *g_dlr = (double *)malloc(basis_size * sizeof(double));
    int32_t from_ir_input_dims[1] = {static_cast<int32_t>(basis_size)};
    int status_from_IR = spir_ir_to_dlr_dd(dlr, SPIR_ORDER_COLUMN_MAJOR, ndim,
                                          from_ir_input_dims, target_dim, Gl, g_dlr);
    REQUIRE(status_from_IR == SPIR_COMPUTATION_SUCCESS);


    // Get Matsubara basis functions
    int32_t uhat_status;
    spir_funcs *uhat = spir_basis_get_uhat(dlr, &uhat_status);
    REQUIRE(uhat_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(uhat != nullptr);

    c_complex *giv_ref = (c_complex *)malloc(n_smpl_points * sizeof(c_complex));
    int32_t smpl_input_dims[1] = {basis_size};
    int32_t status_eval = spir_sampling_evaluate_dz(
        smpl, SPIR_ORDER_COLUMN_MAJOR, ndim,
        smpl_input_dims, target_dim, Gl, giv_ref
    );
    REQUIRE(status_eval == SPIR_COMPUTATION_SUCCESS);

    // Evaluate DLR basis functions at Matsubara frequencies
    c_complex *basis_values = (c_complex *)malloc(n_smpl_points * basis_size * sizeof(c_complex));
    int32_t status_eval_for_dlr = spir_funcs_evaluate_matsubara(
        uhat, SPIR_ORDER_COLUMN_MAJOR, n_smpl_points,
        smpl_points, basis_values
    );
    REQUIRE(status_eval_for_dlr == SPIR_COMPUTATION_SUCCESS);

    // Compute the sum of basis functions weighted by coefficients
    c_complex *giv = (c_complex *)malloc(n_smpl_points * sizeof(c_complex));
    for (int i = 0; i < n_smpl_points; i++) {
        giv[i] = 0;
        for (int j = 0; j < basis_size; j++) {
            giv[i] += g_dlr[j] * basis_values[j * n_smpl_points + i];
        }
    }

    // Clean up
    free(basis_values);
    spir_funcs_destroy(uhat);

    for (int i = 0; i < n_smpl_points; i++) {
        // Compare real and imaginary parts with appropriate tolerance
        double dzr = (__real__(giv_ref[i]) - __real__(giv[i]));
        double dzi = (__imag__(giv_ref[i]) - __imag__(giv[i]));
        double dz = std::sqrt(dzr * dzr + dzi * dzi);
        REQUIRE(dz < 300 * epsilon);
    }
    free(Gl);
    free(g_dlr);
    free(giv_ref);
    free(giv);
    free(smpl_points);
    spir_basis_destroy(basis);
    spir_basis_destroy(dlr);
    spir_basis_destroy(dlr_with_poles);
}

TEST_CASE("DiscreteLehmannRepresentation", "[cinterface]")
{
    SECTION("DiscreteLehmannRepresentation Constructor Fermionic")
    {
        test_finite_temp_basis_dlr<sparseir::Fermionic>();
    }

    SECTION("DiscreteLehmannRepresentation Constructor Bosonic")
    {
        test_finite_temp_basis_dlr<sparseir::Bosonic>();
    }
}
*/