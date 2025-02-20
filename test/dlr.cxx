#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <memory>
#include <random>
#include <sparseir/sparseir-header-only.hpp>

using namespace sparseir;
using Catch::Approx;

TEST_CASE("DLR Tests", "[dlr]")
{
    SECTION("Compression Tests")
    {
        // Test parameters
        const double beta = 10000.0;
        const double omega_max = 1.0;
        const double epsilon = 1e-12;

        LogisticKernel kernel(beta * omega_max);
        auto sve_result = compute_sve(kernel, epsilon);

        auto test_statistics = [&](auto stat) {
            using Statistics = decltype(stat);
            using BasisType = FiniteTempBasis<Statistics>;

            auto basis = std::make_shared<BasisType>(
                beta, omega_max, epsilon, kernel, sve_result);
            auto dlr = DiscreteLehmannRepresentation<Statistics, BasisType>(*basis);

            // Random generation
            std::mt19937 gen(982743);
            std::uniform_real_distribution<> dis(0.0, 1.0);

            const int num_poles = 10;
            Eigen::VectorXd poles(num_poles);
            Eigen::VectorXd coeffs(num_poles);
            for(int i = 0; i < num_poles; ++i) {
                poles(i) = omega_max * (2.0 * dis(gen) - 1.0);
                coeffs(i) = 2.0 * dis(gen) - 1.0;
            }

            REQUIRE(poles.array().abs().maxCoeff() <= omega_max);

            auto dlr_poles = DiscreteLehmannRepresentation<Statistics, BasisType>(*basis, poles);
            auto Gl = dlr_poles.to_IR(coeffs);
            auto g_dlr = dlr.from_IR(Gl);

            // Test on Matsubara frequencies
            auto smpl = MatsubaraSampling<Statistics>(basis);
            auto smpl_for_dlr = MatsubaraSampling<Statistics>(
                std::make_shared<decltype(dlr)>(dlr),
                smpl.sampling_points());

            auto giv_ref = smpl.evaluate(Gl);
            auto giv = smpl_for_dlr.evaluate(g_dlr);

            REQUIRE((giv - giv_ref).array().abs().maxCoeff() <= 300 * epsilon);
        };

        test_statistics(Fermionic{});
        test_statistics(Bosonic{});
    }

    SECTION("Boson Specific Tests")
    {
        double beta = 2.0;
        double omega_max = 21.0;
        double epsilon = 1e-7;

        LogisticKernel kernel(beta * omega_max);
        auto sve_result = compute_sve(kernel, epsilon);

        auto basis_b = std::make_shared<FiniteTempBasis<Bosonic>>(
            beta, omega_max, epsilon, kernel, sve_result);

        // Test specific coefficients and poles
        Eigen::Vector2d coeff(1.1, 2.0);
        Eigen::Vector2d omega_p(2.2, -1.0);

        auto rho_l_pole = basis_b->v(omega_p) * coeff;
        auto gl_pole = -basis_b->s.array() * rho_l_pole.array();

        auto sp = DiscreteLehmannRepresentation(*basis_b, omega_p);
        auto gl_pole2 = sp.to_IR(coeff);

        REQUIRE((gl_pole - gl_pole2).array().abs().maxCoeff() <= 300 * epsilon);
    }

    SECTION("MatsubaraPoles Tests")
    {
        double beta = M_PI;
        Eigen::Vector3d poles(2.0, 3.3, 9.3);

        // Test Fermionic poles
        {
            MatsubaraPoles<Fermionic> mpb(beta, poles);
            std::vector<int> n;
            for(int i = -12345; i <= 987; i += 2) {
                n.push_back(i);
            }
            auto result = mpb(n);

            for(size_t i = 0; i < n.size(); ++i) {
                for(size_t j = 0; j < poles.size(); ++j) {
                    std::complex<double> expected(0.0, valueim<Fermionic>(n[i], beta));
                    expected = 1.0 / (expected - poles[j]);
                    REQUIRE(std::abs(result(j, i) - expected) < 1e-12);
                }
            }
        }

        // Test Bosonic poles
        {
            MatsubaraPoles<Bosonic> mbp(beta, poles);
            std::vector<int> n;
            for(int i = -234; i <= 13898; i += 2) {
                n.push_back(i);
            }
            auto result = mbp(n);

            for(size_t i = 0; i < n.size(); ++i) {
                for(size_t j = 0; j < poles.size(); ++j) {
                    std::complex<double> expected(0.0, valueim<Bosonic>(n[i], beta));
                    expected = std::tanh(beta * poles[j] / 2.0) / (expected - poles[j]);
                    REQUIRE(std::abs(result(j, i) - expected) < 1e-12);
                }
            }
        }
    }
}