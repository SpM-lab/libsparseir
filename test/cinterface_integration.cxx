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

template <typename S, typename K>
void integration_test(double beta, double lambda, double epsilon, int ndim, int dim)
{
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

}