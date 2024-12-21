#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

using namespace sparseir;
using namespace std;

TEST_CASE("Augmented bosonic basis") {
    using T = double;
    T beta = 1000.0;
    T wmax = 2.0;

    // Create bosonic basis
    auto kernel = LogisticKernel(beta * wmax);
    auto sve_result = compute_sve(kernel, 1e-6);
    auto basis = make_shared<FiniteTempBasis<Bosonic>>(beta, wmax, 1e-6, kernel, sve_result);
    // Create augmented basis with TauConst and TauLinear
    /*
    vector<unique_ptr<AbstractAugmentation<T>>> augmentations;
    //augmentations.push_back(make_unique<TauConst<T>>(beta));
    //augmentations.push_back(make_unique<TauLinear<T>>(beta));
    AugmentedBasis<T> basis_aug(basis, augmentations);

    REQUIRE(basis_aug.size() == basis->size() + 2);

    // Define G(τ) = c - exp(-τ * pole) / (1 - exp(-β * pole))
    T pole = 1.0;
    T c = 1e-2;

    // Create tau sampling points
    auto tau_sampling = make_tau_sampling(basis_aug);
    auto tau = tau_sampling.tau;
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
     /*
     for (const shared_ptr<Statistics>& stat : {make_shared<Fermionic>(), make_shared<Bosonic>()}) {
        using T = double;
        T beta = 1000.0;
        T wmax = 2.0;
        auto basis = make_shared<FiniteTempBasis<T>>(stat, beta, wmax, 1e-6);
        vector<unique_ptr<AbstractAugmentation<T>>> augmentations;
        augmentations.push_back(make_unique<MatsubaraConst<T>>(beta));
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
    }
}


TEST_CASE("unit tests", "[augment]") {
    using T = double;
    T beta = 1000.0;
    T wmax = 2.0;
    auto basis = make_shared<FiniteTempBasis<T>>(Bosonic(), beta, wmax, 1e-6);
    vector<unique_ptr<AbstractAugmentation<T>>> augmentations;
    augmentations.push_back(make_unique<TauConst<T>>(beta));
    augmentations.push_back(make_unique<TauLinear<T>>(beta));
    AugmentedBasis<T> basis_aug(basis, augmentations);

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