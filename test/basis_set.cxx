#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>


#include <memory>
#include <complex>

using namespace sparseir;

TEST_CASE("FiniteTempBasisSet consistency tests", "[basis_set]") {

    SECTION("Consistency") {
        // Define parameters
        double beta = 1.0;          // Inverse temperature
        double omega_max = 10.0;    // Maximum frequency
        double epsilon = 1e-5;      // Desired accuracy

        // Create kernels
        sparseir::LogisticKernel kernel;

        // Create shared_ptr instances of FiniteTempBasis
        auto basis_f = std::make_shared<sparseir::FiniteTempBasis<sparseir::Fermionic>>(
            beta, omega_max, epsilon, kernel);
        auto basis_b = std::make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
            beta, omega_max, epsilon, kernel);

        // Create TauSampling instances
        //sparseir::TauSampling<double> smpl_tau_f(basis_f);
        //sparseir::TauSampling<double> smpl_tau_b(basis_b);
        //TauSampling<double> smpl_tau_f(basis_f);
        //auto basis_b = std::make_shared<FiniteTempBasis<Bosonic>>(/* constructor arguments */);
        //TauSampling<double> smpl_tau_b(basis_b);
        /*
        // Create TauSampling objects
        TauSampling<double> smpl_tau_f(basis_f);
        TauSampling<double> smpl_tau_b(basis_b);

        // Create MatsubaraSampling objects
        MatsubaraSampling<std::complex<double>> smpl_wn_f(basis_f);
        MatsubaraSampling<std::complex<double>> smpl_wn_b(basis_b);

        // Create FiniteTempBasisSet
        FiniteTempBasisSet<double> bs(beta, omega_max, epsilon, sve_result);

        // Check that sampling points are equal
        REQUIRE(smpl_tau_f.sampling_points().isApprox(smpl_tau_b.sampling_points()));
        REQUIRE(smpl_tau_f.sampling_points().isApprox(bs.tau()));

        // Check that matrices are equal
        REQUIRE(smpl_tau_f.matrix().isApprox(smpl_tau_b.matrix()));
        REQUIRE(bs.smpl_tau_f()->matrix().isApprox(smpl_tau_f.matrix()));
        REQUIRE(bs.smpl_tau_b()->matrix().isApprox(smpl_tau_b.matrix()));

        REQUIRE(bs.smpl_wn_f()->matrix().isApprox(smpl_wn_f.matrix()));
        REQUIRE(bs.smpl_wn_b()->matrix().isApprox(smpl_wn_b.matrix()));
        */
    }
}