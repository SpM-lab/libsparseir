#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <catch2/catch_test_macros.hpp>

#include <catch2/catch_approx.hpp> // for Approx
#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

#include <random>
#include <complex>
#include <memory>
#include <vector>
#include <string>

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


// A helper function template to test both Bosonic and Fermionic in one place
template <typename Stat>
void test_fit_from_tau_for_stat()
{
    double beta = 1.0;
    vector<double> lambdas = {10.0, 42.0};

    for (auto Lambda : lambdas) {
        SECTION("Testing with Λ=" + std::to_string(Lambda) + " for "
                + (std::is_same<Stat, Bosonic>::value ? "Bosonic" : "Fermionic"))
        {
            double wmax = Lambda / beta;
            auto kernel = LogisticKernel(wmax);
            auto sve_result = compute_sve(kernel, 1e-15);

            // Create a basis matching the template statistic
            auto basis = make_shared<FiniteTempBasis<Stat>>(
                beta, wmax, 1e-15, kernel, sve_result);

            REQUIRE(basis->size() > 0);  // Check basis size

            // Create TauSampling with the same statistic
            auto tau_sampling = make_shared<TauSampling<Stat>>(basis);

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

            // Fill rhol_tensor with random complex values
            for (int i = 0; i < rhol_tensor.size(); ++i) {
                rhol_tensor.data()[i] = ComplexF64(dis(gen), dis(gen));
            }

            // Convert basis->s (real) to a complex tensor for elementwise ops
            Eigen::VectorXd s_vector = basis->s;
            Eigen::Tensor<ComplexF64, 1> s_tensor(s_vector.size());
            for (Eigen::Index i = 0; i < s_vector.size(); ++i) {
                s_tensor(i) = ComplexF64(s_vector(i), 0.0);
            }

            // Reshape and broadcast s_tensor to match rhol_tensor dimensions
            Eigen::array<Eigen::Index, 4> new_shape = {s_size, 1, 1, 1};
            Eigen::Tensor<ComplexF64, 4> s_reshaped = s_tensor.reshape(new_shape);
            Eigen::array<Eigen::Index, 4> bcast = {1, d1, d2, d3};

            // originalgl = -s_reshaped * rhol_tensor
            Eigen::Tensor<ComplexF64, 4> originalgl =
                (-s_reshaped.broadcast(bcast)) * rhol_tensor;

            // Test evaluate() and fit() along each dimension
            for (int dim = 0; dim < 4; ++dim) {
                // Move the "frequency" dimension around
                Eigen::Tensor<ComplexF64, 4> gl = movedim(originalgl, 0, dim);

                // Evaluate from real-time/tau to imaginary-time/tau
                Eigen::Tensor<ComplexF64, 4> gtau = tau_sampling->evaluate(gl, dim);

                // Check shapes
                REQUIRE(gtau.dimension(0) == gl.dimension(0));
                REQUIRE(gtau.dimension(1) == gl.dimension(1));
                REQUIRE(gtau.dimension(2) == gl.dimension(2));
                REQUIRE(gtau.dimension(3) == gl.dimension(3));

                // Fit back to original
                Eigen::Tensor<ComplexF64, 4> gl_from_tau = tau_sampling->fit(gtau, dim);

                // Check shapes again
                REQUIRE(gl_from_tau.dimension(0) == gl.dimension(0));
                REQUIRE(gl_from_tau.dimension(1) == gl.dimension(1));
                REQUIRE(gl_from_tau.dimension(2) == gl.dimension(2));
                REQUIRE(gl_from_tau.dimension(3) == gl.dimension(3));

                // Numerical check
                REQUIRE(sparseir::tensorIsApprox(gl_from_tau, gl, 1e-10));
            }
        }
    }
}

// Now just call the helper function for both Bosonic and Fermionic
TEST_CASE("fit from tau for both statistics, Λ in {10, 42}", "[sampling]")
{
    test_fit_from_tau_for_stat<Bosonic>();
    test_fit_from_tau_for_stat<Fermionic>();
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

TEST_CASE("tau noise with stat (Bosonic or Fermionic), Λ = 10", "[sampling]")
{
    // Common test logic in a helper lambda
    auto run_noise_test = [&](auto stat_tag) {
        using S = decltype(stat_tag);

        double beta = 1.0;
        double wmax = 10.0;
        double Lambda = beta * wmax;

        // Build kernel and SVE
        auto kernel = sparseir::LogisticKernel(Lambda);
        auto sve_result = sparseir::compute_sve(kernel, 1e-15);

        // Create finite-temperature basis and TauSampling for the given
        // statistic
        auto basis = std::make_shared<sparseir::FiniteTempBasis<S>>(
            beta, wmax, 1e-15, kernel, sve_result);
        auto tau_sampling = std::make_shared<sparseir::TauSampling<S>>(basis);

        // Prepare test data
        Eigen::MatrixXd out = basis->v(Eigen::Vector3d(-0.999, -0.01, 0.5));
        Eigen::VectorXd rhol = out * Eigen::Vector3d(0.8, -0.2, 0.5);
        Eigen::VectorXd Gl_ = basis->s.array() * (rhol.array());

        Eigen::Tensor<double, 1> Gl(Gl_.size());
        for (Eigen::Index i = 0; i < Gl_.size(); ++i) {
            Gl(i) = Gl_(i);
        }
        double Gl_magn = Gl_.norm();

        // Evaluate
        Eigen::Tensor<double, 1> Gτ = tau_sampling->evaluate(Gl);

        // Compute norm
        double Gτ_norm = 0.0;
        for (Eigen::Index i = 0; i < Gτ.size(); ++i) {
            Gτ_norm += Gτ(i) * Gτ(i);
        }
        Gτ_norm = std::sqrt(Gτ_norm);

        // Add noise
        double noise = 1e-5;
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        Eigen::Tensor<double, 1> Gτ_n(Gτ.size());
        for (Eigen::Index i = 0; i < Gτ.size(); ++i) {
            Gτ_n(i) = Gτ(i) + noise * Gτ_norm * distribution(generator);
        }

        // Fit back
        Eigen::Tensor<double, 1> Gl_n = tau_sampling->fit(Gτ_n);

        REQUIRE(sparseir::tensorIsApprox(Gl_n, Gl, 12 * noise * Gl_magn));
    };

    SECTION("Bosonic") { run_noise_test(sparseir::Bosonic{}); }

    SECTION("Fermionic") { run_noise_test(sparseir::Fermionic{}); }
}

TEST_CASE("iω noise with Lambda = 10", "[sampling]")
{
    auto run_noise_test = [](auto statistic, bool positive_only) {
        using S = decltype(statistic);
        CAPTURE(positive_only);

        double beta = 1.0;
        double wmax = 10.0;
        double Lambda = beta * wmax;
        auto kernel = sparseir::LogisticKernel(Lambda);
        auto sve_result = sparseir::compute_sve(kernel, 1e-15);
        auto basis = std::make_shared<sparseir::FiniteTempBasis<S>>(
            beta, wmax, 1e-15, kernel, sve_result);

        auto matsu_sampling = std::make_shared<sparseir::MatsubaraSampling<S>>(
            basis, positive_only);

        auto out = basis->v(Eigen::Vector3d(-0.999, -0.01, 0.5));
        auto rhol = out * Eigen::Vector3d(0.8, -0.2, 0.5);
        Eigen::VectorXd Gl_ = basis->s.array() * (rhol.array());
        double Gl_magn = Gl_.norm();

        // Convert Gl_ to a tensor Gℓ
        Eigen::Tensor<double, 1> Gℓ(Gl_.size());
        for (Eigen::Index i = 0; i < Gl_.size(); ++i) {
            Gℓ(i) = Gl_(i);
        }

        auto Giw = matsu_sampling->evaluate(Gℓ);

        double noise = 1e-5;
        double Giw_norm = 0.0;
        for (Eigen::Index i = 0; i < Giw.size(); ++i) {
            Giw_norm +=
                Giw(i).real() * Giw(i).real() + Giw(i).imag() * Giw(i).imag();
        }
        Giw_norm = std::sqrt(Giw_norm);

        // Use fixed seed for reproducibility in Fermionic case
        std::mt19937 generator = std::mt19937(42);
        std::normal_distribution<double> distribution(0.0, 1.0);

        Eigen::Tensor<std::complex<double>, 1> Giwn_n(Giw.size());
        for (Eigen::Index i = 0; i < Giw.size(); ++i) {
            Giwn_n(i) = Giw(i) + noise * Giw_norm * distribution(generator);
        }

        Eigen::Tensor<std::complex<double>, 1> Gℓ_n =
            matsu_sampling->fit(Giwn_n);
        Eigen::Tensor<double, 1> Gℓ_n_real = Gℓ_n.real();

        REQUIRE(sparseir::tensorIsApprox(Gℓ_n_real, Gℓ,
                                         40 * std::sqrt(1 + positive_only) *
                                             noise * Gl_magn));
    };

    SECTION("Bosonic") {
        for (bool positive_only : {true, false}) {
            run_noise_test(sparseir::Bosonic{}, positive_only);
        }
    }

    SECTION("Fermionic") {
        // TODO: support positive_only = true
        // TODO: support positive_only = false
        for (bool positive_only : {true, false}) {
            run_noise_test(sparseir::Fermionic{}, positive_only);
        }
    }
}

TEST_CASE("make_split_svd", "[sampling]")
{
    // Eigen の 3x5 複素数行列 (std::complex<double> 型) を定義
    Eigen::Matrix<std::complex<double>, 3, 5> A;

    // 1行目の初期化
    A(0, 0) = std::complex<double>(0.6151905408205645, 0.6491201503794648);
    A(0, 1) = std::complex<double>(0.06984266107535575, 0.05890187817629955);
    A(0, 2) = std::complex<double>(0.5081681662602489, 0.3938359266698195);
    A(0, 3) = std::complex<double>(0.07965418601180585, 0.09314078427258288);
    A(0, 4) = std::complex<double>(0.5757601025277497, 0.12690424725621652);

    // 2行目の初期化
    A(1, 0) = std::complex<double>(0.7850415647262474, 0.6538533076509034);
    A(1, 1) = std::complex<double>(0.8001878770067667, 0.8714898802836404);
    A(1, 2) = std::complex<double>(0.3850845754724803, 0.5445983221231325);
    A(1, 3) = std::complex<double>(0.56333871776691, 0.04098652756553345);
    A(1, 4) = std::complex<double>(0.5074905499683049, 0.5069257697790468);

    // 3行目の初期化
    A(2, 0) = std::complex<double>(0.49554527311834273, 0.7540215803541244);
    A(2, 1) = std::complex<double>(0.6988196049973501, 0.7871362394492722);
    A(2, 2) = std::complex<double>(0.19625883331364058, 0.2997283029235214);
    A(2, 3) = std::complex<double>(0.6250101801493309, 0.07868380451798473);
    A(2, 4) = std::complex<double>(0.2344187232400199, 0.034048468932398324);

    auto svd = sparseir::make_split_svd(A, false);
    std::vector<double> svals_expected = {
        2.5882898597468316, 0.7807498949326078, 0.5561024337919539,
        0.4302484700046635, 0.07956198767214633};
    for (int i = 0; i < svals_expected.size(); i++) {
        REQUIRE(svd.singularValues()(i) == Approx(svals_expected[i]));
    }

    auto svd_has_zero = sparseir::make_split_svd(A, true);
    std::vector<double> svals_expected_has_zero = {
        2.5111425303908983, 0.7129749569362869, 0.5560724784271746,
        0.2813955280177761, 0.04717410206795309};
    for (int i = 0; i < svals_expected_has_zero.size(); i++) {
        REQUIRE(svd_has_zero.singularValues()(i) ==
                Approx(svals_expected_has_zero[i]));
    }
}
