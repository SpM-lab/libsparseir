#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

#include <random>
#include <complex>
#include <memory>

using namespace sparseir;
using namespace std;

using ComplexF64 = std::complex<double>;

TEST_CASE("Sampling Tests") {
    double beta = 1.0;
    vector<double> lambdas = {10.0, 42.0};

    for (auto Lambda : lambdas) {
        SECTION("Testing with Λ=" + std::to_string(Lambda)) {
            auto kernel = LogisticKernel(beta * Lambda);
            auto sve_result = compute_sve(kernel, 1e-15);
            auto basis = make_shared<FiniteTempBasis<Bosonic>>(
                beta, Lambda, 1e-15, kernel, sve_result);

            REQUIRE(basis->size() > 0);  // Check basis size

            auto tau_sampling = make_shared<TauSampling<Bosonic>>(basis);

            // Check sampling points
            REQUIRE(tau_sampling->sampling_points().size() > 0);

            // Generate random coefficients of correct size
            Eigen::VectorXcd rhol = Eigen::VectorXcd::Random(basis->size());
            REQUIRE(rhol.size() == basis->size());

            const Eigen::Index s_size = basis->size();
            const Eigen::Index d1 = 2;
            const Eigen::Index d2 = 3;
            const Eigen::Index d3 = 4;
            Eigen::Tensor<ComplexF64, 4> rhol_tensor(s_size, d1, d2, d3);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0, 1);

            for (int i = 0; i < rhol_tensor.size(); ++i) {
                rhol_tensor.data()[i] = ComplexF64(dis(gen), dis(gen));
            }
            Eigen::VectorXd s_vector = basis->s; // Assuming basis->s is Eigen::VectorXd
            Eigen::Tensor<ComplexF64, 1> s_tensor(s_vector.size());

            for (Eigen::Index i = 0; i < s_vector.size(); ++i) {
                s_tensor(i) = ComplexF64(s_vector(i), 0.0); // Assuming real to complex conversion
            }
            Eigen::array<Eigen::Index, 4> new_shape = {s_size, 1, 1, 1};
            Eigen::Tensor<ComplexF64, 4> s_reshaped = s_tensor.reshape(new_shape);
            Eigen::array<Eigen::Index, 4> bcast = {1, d1, d2, d3};
            Eigen::Tensor<ComplexF64, 4> originalgl = (-s_reshaped.broadcast(bcast)) * rhol_tensor;

            for (int dim = 0; dim < 4; ++dim) {
                Eigen::Tensor<ComplexF64, 4> gl = movedim(originalgl, 0, dim);
                Eigen::Tensor<ComplexF64, 4> gtau = tau_sampling->evaluate(gl, dim);

                std::cout << "dim: " << dim << std::endl;
                std::cout << "gtau.dimensions(): " << gtau.dimensions() << std::endl;
                std::cout << "gl.dimensions(): " << gl.dimensions() << std::endl;

                REQUIRE(gtau.dimension(0) == gl.dimension(0));
                REQUIRE(gtau.dimension(1) == gl.dimension(1));
                REQUIRE(gtau.dimension(2) == gl.dimension(2));
                REQUIRE(gtau.dimension(3) == gl.dimension(3));
                Eigen::Tensor<ComplexF64, 4> gl_from_tau = tau_sampling->fit(gtau, dim);
                //REQUIRE(gl_from_tau.isApprox(originalgl, 1e-10));
            }

            // Test evaluate and fit
            /*Eigen::VectorXd gtau = tau_sampling->evaluate(rhol);
            REQUIRE(gtau.size() == tau_sampling->sampling_points().size());

            Eigen::VectorXd rhol_recovered = tau_sampling->fit(gtau);
            REQUIRE(rhol_recovered.size() == rhol.size());
            */
            //REQUIRE(rhol.isApprox(rhol_recovered, 1e-10));
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