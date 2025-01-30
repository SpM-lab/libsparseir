#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

#include <random>
#include <complex>
#include <memory>

using namespace sparseir;
using namespace std;

TEST_CASE("Sampling Tests") {
    double beta = 1.0;
    vector<double> lambdas = {10.0, 42.0};

    for (auto Lambda : lambdas) {
        SECTION("Testing with Λ") {
            auto kernel = LogisticKernel(beta * Lambda);
            auto sve_result = compute_sve(kernel, 1e-15);
            auto basis = make_shared<FiniteTempBasis<Bosonic>>(
                beta, Lambda, 1e-15, kernel, sve_result);

            REQUIRE(basis->size() > 0);  // Check basis size

            auto tau_sampling = make_shared<TauSampling<Bosonic>>(basis);

            // Check sampling points
            REQUIRE(tau_sampling->sampling_points().size() > 0);

            // Generate random coefficients of correct size
            Eigen::VectorXd rhol = Eigen::VectorXd::Random(basis->size());
            REQUIRE(rhol.size() == basis->size());

            // Test evaluate and fit
            Eigen::VectorXd gtau = tau_sampling->evaluate(rhol);
            REQUIRE(gtau.size() == tau_sampling->sampling_points().size());

            Eigen::VectorXd rhol_recovered = tau_sampling->fit(gtau);
            REQUIRE(rhol_recovered.size() == rhol.size());
            REQUIRE(rhol.isApprox(rhol_recovered, 1e-10));
        }
    }
}

TEST_CASE("Matsubara Sampling Tests") {
    double beta = 1.0;
    vector<double> lambdas = {10.0, 42.0};
    vector<bool> positive_only_options = {false, true};

    for (auto Lambda : lambdas) {
        for (bool positive_only : positive_only_options) {
            SECTION("Testing with Λ and positive_only") {
                auto kernel = LogisticKernel(beta * Lambda);
                auto sve_result = compute_sve(kernel, 1e-15);
                auto basis = make_shared<FiniteTempBasis<Bosonic>>(
                    beta, Lambda, 1e-15, kernel, sve_result);

                /*
                auto matsu_sampling = make_shared<MatsubaraSampling<double>>(basis, positive_only);

                Eigen::VectorXd rhol = Eigen::VectorXd::Random(basis->size());
                Eigen::VectorXd gl = basis->s.array() * (-rhol.array());

                Eigen::VectorXcd giw = matsu_sampling->evaluate(gl);
                Eigen::VectorXd gl_from_iw = matsu_sampling->fit(giw);
                REQUIRE(gl.isApprox(gl_from_iw, 1e-6));
                */
            }
        }
    }
}

TEST_CASE("Conditioning Tests") {
    double beta = 3.0;
    double wmax = 3.0;
    double epsilon = 1e-6;

    auto kernel = LogisticKernel(beta * wmax);
    auto sve_result = compute_sve(kernel, epsilon);
    auto basis = make_shared<FiniteTempBasis<Bosonic>>(
        beta, wmax, epsilon, kernel, sve_result);

    /*
    auto tau_sampling = make_shared<TauSampling<Bosonic>>(basis);
    auto matsu_sampling = make_shared<MatsubaraSampling<Bosonic>>(basis);

    double cond_tau = tau_sampling->cond();
    double cond_matsu = matsu_sampling->cond();

    REQUIRE(cond_tau < 3.0);
    REQUIRE(cond_matsu < 5.0);
    */
}

TEST_CASE("Error Handling Tests") {
    double beta = 3.0;
    double wmax = 3.0;
    double epsilon = 1e-6;

    auto kernel = LogisticKernel(beta * wmax);
    auto sve_result = compute_sve(kernel, epsilon);
    auto basis = make_shared<FiniteTempBasis<Bosonic>>(
        beta, wmax, epsilon, kernel, sve_result);

    /*
    auto tau_sampling = make_shared<TauSampling<Bosonic>>(basis);
    auto matsu_sampling = make_shared<MatsubaraSampling<double>>(basis);

    Eigen::VectorXd incorrect_size_vec(100);

    REQUIRE_THROWS_AS(tau_sampling->evaluate(incorrect_size_vec), std::invalid_argument);
    REQUIRE_THROWS_AS(tau_sampling->fit(incorrect_size_vec), std::invalid_argument);
    REQUIRE_THROWS_AS(matsu_sampling->evaluate(incorrect_size_vec), std::invalid_argument);
    REQUIRE_THROWS_AS(matsu_sampling->fit(incorrect_size_vec), std::invalid_argument);
    */
}