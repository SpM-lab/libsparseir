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

    for (int i = 0; i < basis_size; i++) {
        std::cout << "Gl[" << i << "] = " << Gl[i] << std::endl;
    }

    REQUIRE(status_to_IR == SPIR_COMPUTATION_SUCCESS);
    double *g_dlr = (double *)malloc(basis_size * sizeof(double));
    int32_t from_ir_input_dims[1] = {static_cast<int32_t>(basis_size)};
    int status_from_IR = spir_dlr_from_IR(dlr, SPIR_ORDER_COLUMN_MAJOR, ndim,
                                          from_ir_input_dims, target_dim, Gl, g_dlr);
    REQUIRE(status_from_IR == SPIR_COMPUTATION_SUCCESS);

    for (int i = 0; i < basis_size; i++) {
        std::cout << "g_dlr[" << i << "] = " << g_dlr[i] << std::endl;
    }

    spir_sampling *smpl;
    bool positive_only = false;
    int32_t smpl_status = spir_matsubara_sampling_new(&smpl, basis, positive_only);
    REQUIRE(smpl_status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(smpl != nullptr);
    double *g_dlr_smpl = (double *)malloc(basis_size * sizeof(double));
    int32_t smpl_input_dims[1] = {basis_size};
    int32_t status_eval = spir_sampling_evaluate_dd(smpl, SPIR_ORDER_COLUMN_MAJOR, ndim, smpl_input_dims, target_dim, g_dlr, g_dlr_smpl);
    REQUIRE(status_eval == SPIR_COMPUTATION_SUCCESS);

    for (int i = 0; i < basis_size; i++) {
        std::cout << "g_dlr_smpl[" << i << "] = " << g_dlr_smpl[i] << std::endl;
    }

    free(Gl);
    free(g_dlr);

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