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


template <typename S>
void test_finite_temp_basis_dlr()
{
    const double beta = 10000.0;
    const double wmax = 1.0;
    const double epsilon = 1e-12;

    auto stat = get_stat<S>();

    spir_finite_temp_basis *basis =
        spir_finite_temp_basis_new(stat, beta, wmax, epsilon);
    REQUIRE(basis != nullptr);

    spir_dlr *dlr = spir_dlr_new(basis);
    REQUIRE(dlr != nullptr);

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

    spir_dlr *dlr_with_poles =
        spir_dlr_new_with_poles(basis, npoles, poles.data());
    REQUIRE(dlr_with_poles != nullptr);
    int fitmat_rows = spir_dlr_fitmat_rows(dlr_with_poles);
    int fitmat_cols = spir_dlr_fitmat_cols(dlr_with_poles);
    REQUIRE(fitmat_rows >= 0);
    REQUIRE(fitmat_cols == npoles);
    double *Gl = (double *)malloc(fitmat_rows * sizeof(double));
    int32_t to_ir_input_dims[1] = {npoles};
    int status_to_IR = spir_dlr_to_IR(dlr_with_poles, SPIR_ORDER_COLUMN_MAJOR,
                                      1, to_ir_input_dims, coeffs.data(), Gl);

    REQUIRE(status_to_IR == SPIR_COMPUTATION_SUCCESS);
    double *g_dlr = (double *)malloc(fitmat_rows * sizeof(double));
    int32_t from_ir_input_dims[1] = {static_cast<int32_t>(fitmat_rows)};
    int status_from_IR = spir_dlr_from_IR(dlr, SPIR_ORDER_COLUMN_MAJOR, 1,
                                          from_ir_input_dims, Gl, g_dlr);
    REQUIRE(status_from_IR == SPIR_COMPUTATION_SUCCESS);

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