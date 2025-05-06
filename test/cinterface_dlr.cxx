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
template <typename Statistics>
void test_finite_temp_basis_dlr()
{
    const double beta = 10000.0;
    const auto stat = get_stat<Statistics>();
    const double wmax = 1.0;
    const double epsilon = 1e-12;

    spir_finite_temp_basis *basis;
    int32_t basis_status = spir_finite_temp_basis_new(&basis, stat, beta, wmax, epsilon);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);
    int32_t basis_size;
    basis_status = spir_finite_temp_basis_get_size(basis, &basis_size);
    REQUIRE(basis_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis_size >= 0);


    spir_dlr *dlr;
    int32_t dlr_status = spir_dlr_new(&dlr, basis);
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

    spir_dlr *dlr_with_poles;
    int32_t poles_status = spir_dlr_new_with_poles(&dlr_with_poles, basis, npoles, poles.data());
    REQUIRE(poles_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(dlr_with_poles != nullptr);

    double *Gl = (double *)malloc(basis_size * sizeof(double));
    std::cout << "basis_size = " << basis_size << std::endl;
    int32_t to_ir_input_dims[1] = {npoles};
    int32_t ndim = 1;
    int32_t target_dim = 0;
    int status_to_IR = spir_dlr_to_IR_dd(dlr_with_poles, SPIR_ORDER_COLUMN_MAJOR,
                                      ndim, to_ir_input_dims, target_dim, coeffs.data(), Gl);

    REQUIRE(status_to_IR == SPIR_COMPUTATION_SUCCESS);
    double *g_dlr = (double *)malloc(basis_size * sizeof(double));
    int32_t from_ir_input_dims[1] = {static_cast<int32_t>(basis_size)};
    int status_from_IR = spir_dlr_from_IR_dd(dlr, SPIR_ORDER_COLUMN_MAJOR, ndim,
                                          from_ir_input_dims, target_dim, Gl, g_dlr);
    REQUIRE(status_from_IR == SPIR_COMPUTATION_SUCCESS);

    spir_sampling *smpl;
    bool positive_only = false;
    int32_t smpl_status = spir_matsubara_sampling_new(&smpl, basis, positive_only);
    REQUIRE(smpl_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(smpl != nullptr);

    int32_t n_smpl_points;
    smpl_status = spir_sampling_get_num_points(smpl, &n_smpl_points);
    REQUIRE(smpl_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(n_smpl_points > 0);

    int32_t *smpl_points = (int32_t *)malloc(n_smpl_points * sizeof(int32_t));
    smpl_status = spir_matsubara_sampling_get_sampling_points(smpl, n_smpl_points, smpl_points);
    REQUIRE(smpl_status == SPIR_COMPUTATION_SUCCESS);

    spir_sampling *smpl_for_dlr;
    int32_t smpl_for_dlr_status = spir_matsubara_sampling_dlr_new(&smpl_for_dlr, dlr, n_smpl_points, smpl_points, positive_only);
    REQUIRE(smpl_for_dlr_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(smpl_for_dlr != nullptr);

    c_complex *giv_ref = (c_complex *)malloc(n_smpl_points * sizeof(c_complex));
    int32_t smpl_input_dims[1] = {basis_size};
    int32_t status_eval = spir_sampling_evaluate_dz(
        smpl, SPIR_ORDER_COLUMN_MAJOR, ndim,
        smpl_input_dims, target_dim, Gl, giv_ref
    );
    REQUIRE(status_eval == SPIR_COMPUTATION_SUCCESS);

    int32_t smpl_for_dlr_input_dims[1] = {basis_size};

    c_complex *giv = (c_complex *)malloc(n_smpl_points * sizeof(c_complex));
    int32_t status_eval_for_dlr = spir_sampling_evaluate_dz(
        smpl_for_dlr, SPIR_ORDER_COLUMN_MAJOR, ndim,
        smpl_for_dlr_input_dims, target_dim, g_dlr, giv
    );
    REQUIRE(status_eval_for_dlr == SPIR_COMPUTATION_SUCCESS);

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
    spir_destroy_finite_temp_basis(basis);
    spir_destroy_dlr(dlr);
    spir_destroy_dlr(dlr_with_poles);
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