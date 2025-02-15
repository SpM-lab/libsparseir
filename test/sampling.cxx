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
using namespace std;
using Catch::Approx;

using ComplexF64 = std::complex<double>;

TEST_CASE("TauSampling Constructor Test", "[sampling]") {
    double beta = 1.0;
    double wmax = 10.0;
    auto kernel = LogisticKernel(beta * wmax);
    auto sve_result = compute_sve(kernel, 1e-15);
    auto basis = make_shared<FiniteTempBasis<Bosonic>>(beta, wmax, 1e-15,
                                                           kernel, sve_result);
    TauSampling<Bosonic> tau_sampling(basis);

    // Initialize x vector properly
    Eigen::VectorXd x(1);
    x(0) = 0.3;
    std::vector<double> mat_ref_vec =
        {
            0.8209004724107448,   -0.449271448243545,    -0.8421791851408207,
            1.0907208389572702,   -0.011760907495966977, -1.088150369622618,
            0.9158018684127142,   0.32302607984184495,   -1.181401559255913,
            0.6561675000550108,   0.6371046638581639,    -1.1785150288689699,
            0.32988851105229844,  0.9070585672694699,    -1.0690576391215845,
            -0.03607541047225425, 1.0973174269744128,    -0.8540342088018821,
            -0.4046629597879494,
        };
    Eigen::MatrixXd mat_ref = Eigen::Map<Eigen::MatrixXd>(mat_ref_vec.data(), 1, 19);
    Eigen::MatrixXd mat = eval_matrix(&tau_sampling, basis, x);

    REQUIRE(basis->u[0](0.3) == Approx(0.8209004724107448));

    REQUIRE(basis->u(x).transpose().isApprox(mat));
    REQUIRE(basis->u(x).transpose().isApprox(mat_ref));

    std::vector<double> sampling_points_ref_vec = {
        0.0036884193900212914,
        0.019354981745749233,
        0.04721451082761008,
        0.08670296984028258,
        0.13697417948305657,
        0.19688831490761904,
        0.2650041152730527,
        0.3395880303712605,
        0.4186486136790497,
        0.4999999999999999,
        0.5813513863209503,
        0.6604119696287395,
        0.7349958847269473,
        0.803111685092381,
        0.8630258205169434,
        0.9132970301597174,
        0.9527854891723898,
        0.9806450182542508,
        0.9963115806099787,
    };
    Eigen::VectorXd sampling_points_ref = Eigen::Map<Eigen::VectorXd>(sampling_points_ref_vec.data(), 19);
    Eigen::VectorXd sampling_points = tau_sampling.sampling_points();
    REQUIRE(sampling_points.isApprox(sampling_points_ref));
}

TEST_CASE("Two-dimensional TauSampling test", "[sampling]") {
    /*
    begin
        Λ = 10.0
        basis = FiniteTempBasis{Bosonic}(1., Λ)
        smpl = TauSampling(basis)
        @test issorted(smpl.sampling_points)
        @test length(basis) == 19
        rng = StableRNG(1234)
        rhol = randn(rng, ComplexF64, length(basis), 2)
        rhol_dim1 = copy(rhol)
        originalgl = -basis.s .* rhol
        dim = 1
        gl = SparseIR.movedim(originalgl, 1 => dim)
        @info size(gl)
        gtau = evaluate(smpl, gl; dim)
        gl_from_tau = fit(smpl, gtau; dim)
        @test gl_from_tau ≈ gl
    end
    */

	std::vector<double> realpart = {
		0.3673381180364523,
		-1.1848713677287497,
		-0.48391946525406376,
		0.06522097090711336,
		-1.0267922189469185,
		-0.6223777767550012,
		-0.10283087463623661,
		0.3778715311165532,
		0.29768602322833465,
		-0.04283291561082807,
		-0.32023762887739543,
		-0.6725592268833712,
		0.30960687841311585,
		0.6918672439487823,
		-0.2388920072830823,
		-0.07085756020012013,
		0.12394467504744563,
		0.06391770716516226,
		-0.21605470690182405,
		0.07042077084531267,
		0.346103005235077,
		-0.8477952574715448,
		0.41009784183386416,
		-0.4498709965313266,
		0.5571298873810305,
		-0.8493729815671984,
		0.28980968852603595,
		-1.4307948463146032,
		-0.10912464313287795,
		-0.8673379720040123,
		-0.7154068328086284,
		0.8283531180967313,
		1.1361739656981185,
		1.5057013085833661,
		-1.1224964468847343,
		-0.08537848365211542,
		-0.39945097032202637,
		-0.7575395088688694,
	};

	std::vector<double> imagpart = {
		0.6400304387848424,
		-0.9155147983732648,
		-0.4851817102131511,
		-0.06338297024020532,
		-1.3488815728765613,
		1.1822088047359214,
		-0.288242258176703,
		-0.570468796120823,
		0.4763084746038559,
		0.03736517002507163,
		-0.056876682732794726,
		-0.07795390274119411,
		0.2633149818135351,
		-0.8815048387138952,
		-0.020330775052176554,
		-0.7661526262003154,
		0.750959163544027,
		0.41540477565392087,
		-0.020689571322099413,
		-0.5441660523605145,
		-0.38846669281165874,
		-0.40987741222234014,
		-0.6741059225323395,
		0.33722179006610414,
		-0.685362676824395,
		0.3093643766704389,
		-1.0494934462703636,
		0.6246635430374898,
		-0.7579801809448388,
		0.33721713831039174,
		0.7085145371621182,
		0.06796430834778598,
		0.02923245777176449,
		1.496840869584957,
		0.24400346936417383,
		0.548982897088907,
		-0.2555793942960095,
		0.5433346850470123,
	};

	// Calculate the proper dimensions
	const int rows = 19;
	const int cols = 2;

	// Create complex vector with proper size
	Eigen::VectorXcd rhol_vec(rows * cols);
	for(int i = 0; i < rhol_vec.size(); i++) {
		rhol_vec(i) = std::complex<double>(realpart[i], imagpart[i]);
	}



    double beta = 1.0;
    double wmax = 10.0;
    auto kernel = LogisticKernel(beta * wmax);
    auto sve_result = compute_sve(kernel, 1e-15);

    auto basis = make_shared<FiniteTempBasis<Bosonic>>(beta, wmax, 1e-15,
                                                           kernel, sve_result);

    // Reshape into matrix with explicit dimensions
    Eigen::MatrixXcd rhol = Eigen::Map<Eigen::MatrixXcd>(rhol_vec.data(), rows, cols);
    Eigen::VectorXd s_vector = basis->s;

    Eigen::Tensor<ComplexF64, 2> originalgl(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            originalgl(i, j) = -s_vector(i) * rhol(i, j);
        }
    }

    REQUIRE(originalgl(0, 0).real() == Approx(-0.4636354126249573));
    REQUIRE(originalgl(0, 1).real() == Approx(-0.08888150057161824));
    REQUIRE(originalgl(1, 0).real() == Approx(0.9909776600839586));
    REQUIRE(originalgl(1, 1).real() == Approx(-0.28946631306766496));

    REQUIRE(originalgl(0, 0).imag() == Approx(-0.8078137334745539));
    REQUIRE(originalgl(0, 1).imag() == Approx(0.6868186007247555));
    REQUIRE(originalgl(1, 0).imag() == Approx(0.7656989082310841));
    REQUIRE(originalgl(1, 1).imag() == Approx(0.32489755829020933));

    auto tau_sampling = TauSampling<Bosonic>(basis);
    Eigen::MatrixXd mat = tau_sampling.get_matrix();

    {
        int dim = 0;
        Eigen::Tensor<ComplexF64, 2> gl = movedim(originalgl, 0, dim);

        REQUIRE(sparseir::tensorIsApprox(gl, originalgl, 1e-10));

        Eigen::Tensor<ComplexF64, 2> gtau = tau_sampling.evaluate(gl, dim);
        Eigen::Tensor<ComplexF64, 2> gl_from_tau = tau_sampling.fit(gtau, dim);

        REQUIRE(sparseir::tensorIsApprox(gl_from_tau, gl, 1e-10));
    }

    {
        int dim = 1;
        Eigen::Tensor<ComplexF64, 2> gl = movedim(originalgl, 0, dim);

        REQUIRE(gl(0, 0).real() == originalgl(0, 0).real());
        REQUIRE(gl(0, 1).real() == originalgl(1, 0).real());
        REQUIRE(gl(1, 0).real() == originalgl(0, 1).real());
        REQUIRE(gl(1, 1).real() == originalgl(1, 1).real());

        REQUIRE(gl(0, 0).imag() == originalgl(0, 0).imag());
        REQUIRE(gl(0, 1).imag() == originalgl(1, 0).imag());
        REQUIRE(gl(1, 0).imag() == originalgl(0, 1).imag());
        REQUIRE(gl(1, 1).imag() == originalgl(1, 1).imag());

        Eigen::Tensor<ComplexF64, 2> gtau = tau_sampling.evaluate(gl, dim);

        REQUIRE(gtau.dimension(0) == 2);
        REQUIRE(gtau.dimension(1) == 19);

        REQUIRE(gtau(0, 0).real() == Approx(-2.5051455282279633));
        REQUIRE(gtau(1, 0).real() == Approx(1.4482376589195032));
        REQUIRE(gtau(0, 1).real() == Approx(-2.383126252301789));
        REQUIRE(gtau(1, 1).real() == Approx(1.2115069651613877));

        Eigen::Tensor<ComplexF64, 2> gl_from_tau = tau_sampling.fit(gtau, dim);

        REQUIRE(sparseir::tensorIsApprox(gl_from_tau, gl, 1e-10));
    }
}

TEST_CASE("Sampling Tests") {
    double beta = 1.0;
    vector<double> lambdas = {10.0, 42.0};

    for (auto Lambda : lambdas) {
        SECTION("Testing with Λ=" + std::to_string(Lambda)) {
            double wmax = Lambda / beta;
            auto kernel = LogisticKernel(wmax);
            auto sve_result = compute_sve(kernel, 1e-15);
            auto basis = make_shared<FiniteTempBasis<Bosonic>>(
                beta, wmax, 1e-15, kernel, sve_result);

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

                //std::cout << "dim: " << dim << std::endl;
                //std::cout << "gtau.dimensions(): " << gtau.dimensions() << std::endl;
                //std::cout << "gl.dimensions(): " << gl.dimensions() << std::endl;

                REQUIRE(gtau.dimension(0) == gl.dimension(0));
                REQUIRE(gtau.dimension(1) == gl.dimension(1));
                REQUIRE(gtau.dimension(2) == gl.dimension(2));
                REQUIRE(gtau.dimension(3) == gl.dimension(3));
                Eigen::Tensor<ComplexF64, 4> gl_from_tau = tau_sampling->fit(gtau, dim);

                REQUIRE(gl_from_tau.dimension(0) == gl.dimension(0));
                REQUIRE(gl_from_tau.dimension(1) == gl.dimension(1));
                REQUIRE(gl_from_tau.dimension(2) == gl.dimension(2));
                REQUIRE(gl_from_tau.dimension(3) == gl.dimension(3));

                //std::cout << "gl_from_tau(0, 0, 0, 0) " << gl_from_tau(0, 0, 0, 0) << std::endl;
                //std::cout << "gl_from_tau(0, 0, 0, 1) " << gl_from_tau(0, 0, 0, 1) << std::endl;
                //std::cout << "gl(0, 0, 0, 0) " << gl(0, 0, 0, 0) << std::endl;
                //std::cout << "gl(0, 0, 0, 1) " << gl(0, 0, 0, 1) << std::endl;

                REQUIRE(sparseir::tensorIsApprox(gl_from_tau, gl, 1e-10));
            }

            // Test evaluate and fit
            //Eigen::VectorXd gtau = tau_sampling->evaluate(rhol);
            //REQUIRE(gtau.size() == tau_sampling->sampling_points().size());

            //Eigen::VectorXd rhol_recovered = tau_sampling->fit(gtau);
            //REQUIRE(rhol_recovered.size() == rhol.size());
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
                double wmax = Lambda / beta;
                auto kernel = LogisticKernel(beta * wmax);
                auto sve_result = compute_sve(kernel, 1e-15);
                auto basis = make_shared<FiniteTempBasis<Bosonic>>(
                    beta, wmax, 1e-15, kernel, sve_result);

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