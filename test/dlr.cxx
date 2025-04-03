#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <memory>
#include <random>
#include <sparseir/sparseir.hpp>
#include "sve_cache.hpp"

using Catch::Approx;

TEST_CASE("DLR Tests", "[dlr]") {
    SECTION("Compression Tests")
    {
        // Test parameters
        const double beta = 10000.0;
        const double omega_max = 1.0;
        const double epsilon = 1e-12;

        auto kernel = std::make_shared<sparseir::LogisticKernel>(beta * omega_max);
        auto sve_result = SVECache::get_sve_result(*kernel, epsilon);

        auto test_statistics = [&](auto stat) {
            using Statistics = decltype(stat);

            auto basis =
                std::make_shared<sparseir::FiniteTempBasis<Statistics>>(
                    beta, omega_max, epsilon, std::static_pointer_cast<sparseir::AbstractKernel>(kernel), sve_result);
            auto dlr =
                sparseir::DiscreteLehmannRepresentation<Statistics>(*basis);

            // Random generation
            std::mt19937 gen(982743);
            std::uniform_real_distribution<> dis(0.0, 1.0);

            const int num_poles = 10;
            Eigen::VectorXd poles(num_poles);
            Eigen::VectorXd coeffs(num_poles);
            for (int i = 0; i < num_poles; ++i) {
                poles(i) = omega_max * (2.0 * dis(gen) - 1.0);
                coeffs(i) = 2.0 * dis(gen) - 1.0;
            }

            REQUIRE(poles.array().abs().maxCoeff() <= omega_max);

            auto dlr_poles =
                sparseir::DiscreteLehmannRepresentation<Statistics>(*basis,
                                                                    poles);

            Eigen::Tensor<double, 1> coeffs_as_tensor(coeffs.size());
            for (int i = 0; i < coeffs.size(); ++i) {
                coeffs_as_tensor(i) = coeffs(i);
            }
            Eigen::Tensor<double, 1> Gl_tensor = dlr_poles.to_IR(coeffs_as_tensor);
            auto g_dlr_tensor = dlr.from_IR(Gl_tensor);

            auto smpl = sparseir::MatsubaraSampling<Statistics>(basis);
            auto smpl_points = smpl.sampling_points();
            auto smpl_for_dlr =
                sparseir::MatsubaraSampling<Statistics>(dlr, smpl_points);

            auto giv_ref = smpl.evaluate(Gl_tensor);
            auto giv = smpl_for_dlr.evaluate(g_dlr_tensor);

            // Compare using tensorIsApprox
            sparseir::tensorIsApprox(giv, giv_ref, 300 * epsilon);
        };

        test_statistics(sparseir::Fermionic{});
        test_statistics(sparseir::Bosonic{});
    }

    SECTION("Boson Specific Tests") {
        double beta = 2.0;
        double omega_max = 21.0;
        double epsilon = 1e-7;

        auto kernel = std::make_shared<sparseir::LogisticKernel>(beta * omega_max);
        auto sve_result = SVECache::get_sve_result(*kernel, epsilon);

        auto basis_b = std::make_shared<sparseir::FiniteTempBasis<sparseir::Bosonic>>(
            beta, omega_max, epsilon, std::static_pointer_cast<sparseir::AbstractKernel>(kernel), sve_result);

        // Test specific coefficients and poles
        Eigen::Vector2d coeff(1.1, 2.0);
        Eigen::Vector2d omega_p(2.2, -1.0);

        Eigen::MatrixXd rho_l_pole = (*basis_b->v)(omega_p) * coeff;
        Eigen::ArrayXXd rho_l_pole_array = rho_l_pole.array();
        Eigen::ArrayXd s_array = basis_b->s.array();
        Eigen::MatrixXd gl_pole = (-rho_l_pole_array * s_array.replicate(1, rho_l_pole.cols())).matrix();

        auto sp = sparseir::DiscreteLehmannRepresentation<sparseir::Bosonic>(*basis_b, omega_p);
        Eigen::Tensor<double, 1> coeff_as_tensor(coeff.size());
        for (int i = 0; i < coeff.size(); ++i) {
            coeff_as_tensor(i) = coeff(i);
        }
        Eigen::Tensor<double, 1> gl_pole2_as_tensor = sp.to_IR(coeff_as_tensor);
        // convert to matrix
        Eigen::Map<const Eigen::MatrixXd> gl_pole2(gl_pole2_as_tensor.data(), gl_pole2_as_tensor.dimension(0), 1);

        REQUIRE((gl_pole.array() - gl_pole2.array()).abs().maxCoeff() <= 300 * epsilon);
    }

    SECTION("MatsubaraPoles Fermionic Tests") {
        double beta = M_PI;
        Eigen::Vector3d poles(2.0, 3.3, 9.3);

        // Choose 100 random odd integers between -12345 and 987
        std::vector<int> n;
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-12345, 987);
        while (n.size() < 100) {
            int x = dist(gen);
            if (x % 2 == 1) {
                n.push_back(x);
            }
        }

        REQUIRE(n.size() == 100);
        sparseir::MatsubaraPoles<sparseir::Fermionic> mpb(beta, poles);

        Eigen::MatrixXcd result = mpb(n);

        std::complex<double> im(0.0, 1.0);  // imaginary unit i
        for (int i = 0; i < n.size(); ++i) {
            for (int j = 0; j < poles.size(); ++j) {
                std::complex<double> n_complex = static_cast<double>(n[i]) * im;
                REQUIRE(std::abs(result(j, i) - 1.0 / (n_complex - poles[j])) < 1e-12);
            }
        }
    }

    SECTION("MatsubaraPoles Bosonic Tests") {
        double beta = M_PI;
        Eigen::Vector3d poles(2.0, 3.3, 9.3);

        sparseir::MatsubaraPoles<sparseir::Bosonic> mbp(beta, poles);
        // Choose 100 random even integers between -234 and 13898
        std::vector<int> n;
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-234, 13898);
        while (n.size() < 100) {
            int x = dist(gen);
            if (x % 2 == 0) {
                n.push_back(x);
            }
        }

        Eigen::MatrixXcd result = mbp(n);

        for (size_t i = 0; i < n.size(); ++i) {
            for (size_t j = 0; j < poles.size(); ++j) {
                std::complex<double> im(0.0, 1.0);  // imaginary unit i
                std::complex<double> n_complex = static_cast<double>(n[i]) * im;
                std::complex<double> numerator = std::tanh(M_PI * poles[j] / 2.0);
                REQUIRE(std::abs(result(j, i) - numerator / (n_complex - poles[j])) < 1e-12);
            }
        }
    }
}
