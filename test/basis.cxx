#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

TEST_CASE("FiniteTempBasis consistency tests", "[basis]") {
    SECTION("Basic consistency") {
        double beta = 2.0;
        double omega_max = 5.0;
        double epsilon = 1e-5;

        // Define the kernel
        sparseir::LogisticKernel kernel(beta * omega_max);

        // Specify both template parameters: S and K
        using FermKernel = sparseir::FiniteTempBasis<sparseir::Fermionic, sparseir::LogisticKernel>;
        using BosKernel = sparseir::FiniteTempBasis<sparseir::Bosonic, sparseir::LogisticKernel>;

        std::pair<FermKernel, BosKernel> bases = sparseir::finite_temp_bases(beta, omega_max, epsilon);

        // Ensure FiniteTempBasisSet is properly instantiated with the kernel type
        /*
        sparseir::FiniteTempBasisSet<sparseir::LogisticKernel> bs(beta, omega_max, epsilon, kernel);

        REQUIRE(bases.first->singular_values().size() == bs.basis_f()->singular_values().size());
        REQUIRE(bases.second->singular_values().size() == bs.basis_b()->singular_values().size());

        // Clean up dynamically allocated memory if applicable
        delete bases.first;
        delete bases.second;
        */
    }

    SECTION("Sampling consistency") {
        double beta = 2.0;
        double omega_max = 5.0;
        double epsilon = 1e-5;

        sparseir::LogisticKernel kernel(beta * omega_max);
        // Specify the template argument for SVEResult
        sparseir::SVEResult<sparseir::LogisticKernel> sve_result = sparseir::compute_sve(kernel);

        using FermKernel = sparseir::FiniteTempBasis<sparseir::Fermionic, sparseir::LogisticKernel>;
        using BosKernel = sparseir::FiniteTempBasis<sparseir::Bosonic, sparseir::LogisticKernel>;

        std::pair<FermKernel, BosKernel> bases = sparseir::finite_temp_bases(beta, omega_max, epsilon, sve_result);
        /*
        sparseir::FiniteTempBasisSet<sparseir::LogisticKernel> bs(beta, omega_max, epsilon, sve_result);

        // Check sampling points consistency
        sparseir::TauSampling<sparseir::Fermionic> smpl_tau_f(bases.first);
        sparseir::TauSampling<sparseir::Bosonic> smpl_tau_b(bases.second);

        REQUIRE(smpl_tau_f.sampling_points() == smpl_tau_b.sampling_points());
        REQUIRE(bs.smpl_tau_f()->matrix() == smpl_tau_f.matrix());
        REQUIRE(bs.smpl_tau_b()->matrix() == smpl_tau_b.matrix());

        // Clean up dynamically allocated memory if applicable
        delete bases.first;
        delete bases.second;
        */
    }

    SECTION("Singular value scaling")
    {
        double beta = 1e-3;
        double omega_max = 1e-3;
        double epsilon = 1e-100;

        sparseir::FiniteTempBasis<sparseir::Fermionic, sparseir::LogisticKernel>
            basis(beta, omega_max, epsilon,
                  sparseir::LogisticKernel(beta * omega_max));
        sparseir::SVEResult<sparseir::LogisticKernel> sve =
            sparseir::compute_sve(sparseir::LogisticKernel(beta * omega_max));

        double scale = std::sqrt(beta / 2.0 * omega_max);
        // Ensure the correct function or member is used for singular values
        Eigen::VectorXd scaled_s_eigen = sve.s * scale;
        REQUIRE(basis.s.isApprox(scaled_s_eigen));

        // Access accuracy as a member variable if it's not a function
        REQUIRE(basis.accuracy == sve.s(sve.s.size() - 1) / sve.s[0]);
    }

    SECTION("Rescaling test") {
        double beta = 3.0;
        double omega_max = 4.0;
        double epsilon = 1e-6;

        // Specify both template parameters
        sparseir::FiniteTempBasis<sparseir::Fermionic, sparseir::LogisticKernel> basis(beta, omega_max, epsilon, sparseir::LogisticKernel(beta * omega_max));
        sparseir::FiniteTempBasis<sparseir::Fermionic, sparseir::LogisticKernel> rescaled_basis = basis.rescale(2.0);

        REQUIRE(rescaled_basis.get_wmax() == 6.0);
    }
}
