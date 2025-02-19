#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <catch2/catch_test_macros.hpp>

#include <catch2/catch_approx.hpp> // for Approx
#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

#include <random>
#include <complex>
#include <memory>

using namespace sparseir;

TEST_CASE("DLR Compression Tests", "[dlr]")
{
    // Common test parameters
    const double beta = 10000.0;
    const double omega_max = 1.0;
    const double epsilon = 1e-12;

    // Create SVE result from logistic kernel
    sparseir::LogisticKernel kernel(beta * omega_max);
    auto sve_result = sparseir::compute_sve(kernel, epsilon);

    SECTION("Fermionic Statistics") {
        auto basis = sparseir::FiniteTempBasis<sparseir::Fermionic>(beta, omega_max, epsilon, kernel, sve_result);
        auto dlr = sparseir::DiscreteLehmannRepresentation(basis);

        // Random number generation setup
        std::mt19937 gen(982743);  // Fixed seed for reproducibility
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        // Generate random poles and coefficients
        const int num_poles = 10;
        Eigen::VectorXd poles(num_poles);
        Eigen::VectorXd coeffs(num_poles);

        for(int i = 0; i < num_poles; i++) {
            poles(i) = omega_max * dis(gen);
            coeffs(i) = 2.0 * dis(gen) - 1.0;
        }

        // Verify poles are within bounds
        REQUIRE(poles.array().abs().maxCoeff() <= omega_max);

        // Create DLR with specific poles and convert to IR
        auto dlr_poles = sparseir::DiscreteLehmannRepresentation(basis, poles);
        auto Gl = dlr_poles.to_IR(coeffs);
        auto g_dlr = dlr.from_IR(Gl);

        // Test on Matsubara frequencies
        auto smpl = sparseir::MatsubaraSampling<std::complex<double>>(basis);
        auto smpl_for_dlr = sparseir::MatsubaraSampling<std::complex<double>>(
            dlr, smpl.sampling_points());

        auto giv_ref = smpl.evaluate(Gl);
        auto giv = smpl_for_dlr.evaluate(g_dlr);

        // Compare results with tolerance
        REQUIRE((giv - giv_ref).array().abs().maxCoeff() <= 300 * epsilon);

        // Test on τ points
        auto smpl_tau = sparseir::TauSampling<double>(basis);
        auto gτ = smpl_tau.evaluate(Gl);

        auto smpl_tau_for_dlr = sparseir::TauSampling<double>(dlr);
        auto gτ2 = smpl_tau_for_dlr.evaluate(g_dlr);

        REQUIRE((gτ - gτ2).array().abs().maxCoeff() <= 300 * epsilon);
    }

    SECTION("Bosonic Statistics") {
        auto basis = sparseir::FiniteTempBasis<sparseir::Bosonic>(beta, omega_max, epsilon, kernel, sve_result);

        // Test specific coefficients and poles
        Eigen::Vector2d coeff(1.1, 2.0);
        Eigen::Vector2d omega_p(2.2, -1.0);

        auto rho_l_pole = basis.v(omega_p) * coeff;
        auto gl_pole = -basis.s.asDiagonal() * rho_l_pole;

        auto sp = sparseir::DiscreteLehmannRepresentation(basis, omega_p);
        auto gl_pole2 = sp.to_IR(coeff);

        REQUIRE((gl_pole - gl_pole2).array().abs().maxCoeff() <= 300 * epsilon);
    }
}

TEST_CASE("MatsubaraPoles Unit Tests", "[dlr]") {
    const double beta = M_PI;
    Eigen::Vector3d poles(2.0, 3.3, 9.3);

    SECTION("Fermionic MatsubaraPoles") {
        sparseir::MatsubaraPoles<sparseir::Fermionic> mpb(beta, poles);

        // Generate random Matsubara frequencies
        std::vector<sparseir::FermionicFreq> freqs;
        for(int n = -12345; n <= 987; n += 2) {
            if(n % 2 != 0) freqs.push_back(sparseir::FermionicFreq(n));
        }

        auto result = mpb(freqs);

        // Compare with analytical result
        for(size_t i = 0; i < freqs.size(); i++) {
            std::complex<double> iw(0, sparseir::valueim(freqs[i], beta));
            for(int j = 0; j < poles.size(); j++) {
                REQUIRE(std::abs(result(j,i) - 1.0/(iw - poles(j))) < 1e-10);
            }
        }
    }

    SECTION("Bosonic MatsubaraPoles") {
        sparseir::MatsubaraPoles<sparseir::Bosonic> mbp(beta, poles);

        // Generate random Matsubara frequencies
        std::vector<sparseir::BosonicFreq> freqs;
        for(int n = -234; n <= 13898; n += 2) {
            if(n % 2 == 0) freqs.push_back(sparseir::BosonicFreq(n));
        }

        auto result = mbp(freqs);

        // Compare with analytical result
        for(size_t i = 0; i < freqs.size(); i++) {
            std::complex<double> iw(0, sparseir::valueim(freqs[i], beta));
            for(int j = 0; j < poles.size(); j++) {
                REQUIRE(std::abs(result(j,i) -
                    std::tanh(M_PI/2 * poles(j))/(iw - poles(j))) < 1e-10);
            }
        }
    }
}