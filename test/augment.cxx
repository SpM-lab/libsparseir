#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp> // for Approx

#include <catch2/catch_test_macros.hpp>
#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

using namespace sparseir;
using namespace std;


TEST_CASE("augment.cxx", "[AbstractAugmentation]")
{
    using Catch::Approx;
    using namespace sparseir;

    SECTION("TauConst")
    {
        REQUIRE_THROWS_AS(TauConst(-34), std::domain_error);

        TauConst tc(123);
        REQUIRE(tc.beta == 123.0);

        REQUIRE_THROWS_AS(tc(-3), std::domain_error);
        REQUIRE_THROWS_AS(tc(321), std::domain_error);

        REQUIRE(tc(100) == 1 / std::sqrt(123));
        MatsubaraFreq<Bosonic> freq0(0);
        MatsubaraFreq<Bosonic> freq92(92);
        REQUIRE(tc(freq0) == std::sqrt(123));
        REQUIRE(tc(freq92) == 0.0);
        MatsubaraFreq<Fermionic> freq93(93);
        REQUIRE_THROWS_AS(tc(freq93), std::invalid_argument);
        auto tdc = tc.deriv();
        REQUIRE(tdc(4.2) == 0.0);

        tdc = tc.deriv(0);
        double x = 4.2;
        // Expected tdc == tc
        REQUIRE(tdc(x) == tc(x));
    }

    SECTION("TauLinear")
    {
        REQUIRE_THROWS_AS(TauLinear(-34), std::domain_error);

        TauLinear tl(123);
        REQUIRE(tl.beta == Approx(123.0));

        REQUIRE_THROWS_AS(tl(-3), std::domain_error);
        REQUIRE_THROWS_AS(tl(321), std::domain_error);
        REQUIRE(tl.norm == Approx(std::sqrt(3.0 / 123.0)));
        double tau = 100;
        REQUIRE(tl(tau) == std::sqrt(3.0 / 123.0) * (2.0 / 123. * tau - 1.));

        MatsubaraFreq<Bosonic> freq0(0);
        REQUIRE(tl(freq0) == 0.0);
        MatsubaraFreq<Bosonic> freq92(92);
        // Calculate the expected complex value
        std::complex<double> expected_value = std::sqrt(3. / 123.) * 2. / std::complex<double>(0, 1) * 123. / (92. * M_PI);
        // Get the actual value from the function
        std::complex<double> actual_value = tl(freq92);

        REQUIRE(actual_value.real() == Approx(expected_value.real()));
        REQUIRE(actual_value.imag() == Approx(expected_value.imag()));

        MatsubaraFreq<Fermionic> freq93(93);
        REQUIRE_THROWS_AS(tl(freq93), std::invalid_argument);

        double x = 4.2;
        auto d0tl = tl.deriv(0);
        REQUIRE(d0tl(x) == tl(x));
        auto dtl = tl.deriv();
        REQUIRE(dtl(4.2) ==
                Approx(std::sqrt(3. / 123.) * 2. / 123.));
        auto ddtl = tl.deriv(2);
        REQUIRE(ddtl(4.2) == 0.0);
    }

    SECTION("MatsubaraConst")
    {
        REQUIRE_THROWS_AS(MatsubaraConst(-34), std::domain_error);

        MatsubaraConst mc(123);
        REQUIRE(mc.beta == Approx(123.0));

        REQUIRE_THROWS_AS(mc(-3), std::domain_error);
        REQUIRE_THROWS_AS(mc(321), std::domain_error);

        REQUIRE(std::isnan(mc(100)));
        MatsubaraFreq<Bosonic> freq0(0);
        REQUIRE(mc(freq0) == 1.0);
        MatsubaraFreq<Bosonic> freq92(0);
        REQUIRE(mc(freq92) == 1.0);
        MatsubaraFreq<Fermionic> freq93(93);
        REQUIRE(mc(freq93) == 1.0);

        auto d0mc = mc.deriv(0);
        auto dmc = mc.deriv();
        double x = 4.2;
        // Expected dmc == mc
        REQUIRE(std::isnan(d0mc(x)));
        REQUIRE(std::isnan(mc(x)));
        REQUIRE(std::isnan(dmc(x)));
        REQUIRE(std::isnan(mc(x)));
    }
}

TEST_CASE("Augmented bosonic basis") {
    using T = double;
    T beta = 1000.0;
    T wmax = 2.0;

    // Create bosonic basis
    LogisticKernel kernel(beta * wmax);
    auto sve_result = compute_sve(kernel, 1e-6);
    shared_ptr<FiniteTempBasis<Bosonic>> basis = make_shared<FiniteTempBasis<Bosonic>>(beta, wmax, 1e-6, kernel, sve_result);
    // Create augmented basis with TauConst and TauLinear
    vector<shared_ptr<AbstractAugmentation>> augmentations;
    augmentations.push_back(make_shared<TauConst>(beta));
    augmentations.push_back(make_shared<TauLinear>(beta));
    PiecewiseLegendrePolyVector u = basis->u;
    PiecewiseLegendreFTVector<Bosonic> uhat = basis->uhat;
    using S = FiniteTempBasis<Bosonic>;
    using B = Bosonic;
    using F = PiecewiseLegendrePolyVector;
    using FTBosonic = PiecewiseLegendreFTVector<Bosonic>;

    // Define G(τ) = c - exp(-τ * pole) / (1 - exp(-β * pole))
    T pole = 1.0;
    T c = 1e-2;
    // Create tau sampling points
    /*
    auto tau_sampling = TauSampling(basis_aug);
    auto tau = tau_sampling.tau;
    //REQUIRE(tau.size() == basis_aug.size());
    Eigen::VectorX<T> gtau(tau.size());
    for (size_t i = 0; i < tau.size(); ++i) {
        gtau(i) = c - exp(-tau(i) * pole) / (1 - exp(-beta * pole));
    }
    T magn = gtau.maxCoeff();

    // This illustrates that "naive" fitting is a problem if the fitting matrix
    // is not well-conditioned.
    Eigen::MatrixXd tau_matrix = tau_sampling.matrix;
    Eigen::VectorX<T> gl_fit_bad = tau_matrix.completeOrthogonalDecomposition().solve(gtau);
    Eigen::VectorX<T> gtau_reconst_bad = tau_matrix * gl_fit_bad;
    REQUIRE(!gtau_reconst_bad.isApprox(gtau, 1e-13 * magn));
    REQUIRE(gtau_reconst_bad.isApprox(gtau, 5e-16 * tau_matrix.norm() * magn));
    REQUIRE(tau_matrix.norm() > 1e7);
    REQUIRE(tau_matrix.rows() == basis_aug.size());
    REQUIRE(tau_matrix.cols() == tau.size());

    // Now do the fit properly
    Eigen::VectorX<T> gl_fit = tau_matrix.colPivHouseholderQr().solve(gtau);
    Eigen::VectorX<T> gtau_reconst = tau_matrix * gl_fit;

    REQUIRE(gtau_reconst.isApprox(gtau, 1e-14 * magn));
    */
}


TEST_CASE("Vertex basis with stat = $stat", "[augment]") {
    // Fix: Use proper template parameters for Statistics
    std::vector<std::shared_ptr<Statistics>> stats = {
        std::make_shared<Fermionic>(),
        std::make_shared<Bosonic>()
    };

    for (const auto& stat : stats) {
        double beta = 1000.0;
        double wmax = 2.0;
        /*
        auto basis = make_shared<FiniteTempBasis<T>>(stat, beta, wmax, 1e-6);
        vector<shared_ptr<AbstractAugmentation<T>>> augmentations;
        augmentations.push_back(make_shared<MatsubaraConst<T>>(beta));
        AugmentedBasis<T> basis_aug(basis, augmentations);
        REQUIRE(!basis_aug.uhat.empty());

        // G(iν) = c + 1 / (iν - pole)
        T pole = 1.0;
        T c = 1.0;
        auto matsu_sampling = make_matsubara_sampling(basis_aug);
        Eigen::VectorXcf gi_n(matsu_sampling.wn.size());
        for (size_t i = 0; i < matsu_sampling.wn.size(); ++i) {
            complex<T> iwn(0, matsu_sampling.wn(i));
            gi_n(i) = c + 1.0 / (iwn - pole);
        }
        Eigen::VectorXcf gl = fit(matsu_sampling, gi_n);
        Eigen::VectorXcf gi_n_reconst = evaluate(matsu_sampling, gl);
        REQUIRE(gi_n_reconst.isApprox(gi_n, gi_n.maxCoeff() * 1e-7));
        */
    }
}


TEST_CASE("unit tests", "[augment]") {
    using T = double;
    T beta = 1000.0;
    T wmax = 2.0;
    using S = Bosonic;
    auto basis = make_shared<FiniteTempBasis<S>>(beta, wmax, 1e-6);
    std::vector<std::shared_ptr<AbstractAugmentation>> augmentations;
    augmentations.push_back(make_shared<TauConst>(beta));
    augmentations.push_back(make_shared<TauLinear>(beta));
    AugmentedBasis<S> basis_aug(basis, augmentations);
    /*

    SECTION("getindex") {
        REQUIRE(basis_aug.u.size() == basis_aug.size());
        REQUIRE(basis_aug.u[0]->operator()(0.0) == basis_aug[0](0.0));
        REQUIRE(basis_aug.u[1]->operator()(0.0) == basis_aug[1](0.0));
    }

    size_t len_basis = basis->size();
    size_t len_aug = len_basis + 2;

    REQUIRE(basis_aug.size() == len_aug);
    // REQUIRE(basis_aug.accuracy == basis->accuracy);
    REQUIRE(basis_aug.Lambda() == beta * wmax);
    REQUIRE(basis_aug.wmax() == wmax);

    REQUIRE(basis_aug.u.size() == len_aug);
    REQUIRE(basis_aug.u(0.8).size() == len_aug);
    //REQUIRE(basis_aug.uhat(MatsubaraFreq<T>(4.0)).size() == len_aug);
    REQUIRE(basis_aug.u.minCoeff() == 0.0);
    REQUIRE(basis_aug.u.maxCoeff() == beta);

    //Further tests omitted for brevity,  adapt as needed from Julia code.
    */
}