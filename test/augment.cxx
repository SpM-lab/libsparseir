#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

using namespace sparseir;

TEST_CASE("Augmented bosonic basis") {
    using T = double;
    T beta = 1000.0;
    T wmax = 2.0;

    // Create bosonic basis
    auto kernel = LogisticKernel(beta * wmax);
    auto sve_result = compute_sve(kernel, 1e-6);
    /*
    auto basis = std::make_shared<FiniteTempBasis<T>>(Bosonic(), beta, wmax, 1e-6, kernel, sve_result);
    // Create augmented basis with TauConst and TauLinear
    std::vector<std::unique_ptr<AbstractAugmentation<T>>> augmentations;
    augmentations.push_back(std::make_unique<TauConst<T>>(beta));
    augmentations.push_back(std::make_unique<TauLinear<T>>(beta));
    AugmentedBasis<T> basis_aug(basis, augmentations);

    REQUIRE(basis_aug.size() == basis->size() + 2);

    // Define G(τ) = c - exp(-τ * pole) / (1 - exp(-β * pole))
    T pole = 1.0;
    T c = 1e-2;

    // Create tau sampling points
    auto tau_sampling = make_tau_sampling(basis_aug);
    auto tau = tau_sampling.tau;

    // Evaluate G(τ)
    std::vector<T> g_tau(tau.size());
    for (size_t i = 0; i < tau.size(); ++i) {
        T exp_term = std::exp(-tau[i] * pole);
        T denominator = 1.0 - std::exp(-beta * pole);
        g_tau[i] = c - exp_term / denominator;
    }

    // Fit the data (implement fit function appropriately)
    auto gl_fit = fit(tau_sampling, g_tau);

    // Reconstruct G(τ)
    auto g_tau_reconst = evaluate(tau_sampling, gl_fit);

    // Check if the reconstructed G(τ) is close to the original
    T magn = *std::max_element(g_tau.begin(), g_tau.end(), [](T a, T b) { return std::abs(a) < std::abs(b); });
    T tolerance = 1e-14 * magn;
    for (size_t i = 0; i < g_tau.size(); ++i) {
        REQUIRE(std::abs(g_tau_reconst[i] - g_tau[i]) < tolerance);
    }
    */
}

TEST_CASE("Vertex basis with Fermionic and Bosonic statistics") {
    using T = double;
    T beta = 1000.0;
    T wmax = 2.0;
    /*
    // Test for both Fermionic and Bosonic statistics
    std::vector<std::shared_ptr<Statistics>> stats = {
        std::make_shared<Fermionic>(),
        std::make_shared<Bosonic>()
    };

    for (const auto& stat : stats) {
        // Create basis
        auto kernel = LogisticKernel(beta * wmax);
        auto sve_result = compute_sve(kernel, 1e-6);
        auto basis = std::make_shared<FiniteTempBasis<T>>(*stat, beta, wmax, 1e-6, kernel, sve_result);

        // Create augmented basis with MatsubaraConst
        std::vector<std::unique_ptr<AbstractAugmentation<T>>> augmentations;
        augmentations.push_back(std::make_unique<MatsubaraConst<T>>(beta));
        AugmentedBasis<T> basis_aug(basis, augmentations);

        // Create Matsubara sampling points
        auto matsu_sampling = make_matsubara_sampling(basis_aug);

        // Evaluate G(iν) = c + 1 / (iν - pole)
        T pole = 1.0;
        T c = 1.0;
        std::vector<std::complex<T>> gi_n;
        auto wn = matsu_sampling.wn;
        for (const auto& w : wn) {
            std::complex<T> iwn(0, w);
            gi_n.push_back(c + 1.0 / (iwn - pole));
        }

        // Fit the data
        auto gl_fit = fit(matsu_sampling, gi_n);

        // Reconstruct G(iν)
        auto gi_n_reconst = evaluate(matsu_sampling, gl_fit);

        // Check if the reconstructed G(iν) is close to the original
        T max_abs_gi_n = 0.0;
        for (const auto& val : gi_n) {
            if (std::abs(val) > max_abs_gi_n) {
                max_abs_gi_n = std::abs(val);
            }
        }
        T tolerance = 1e-7 * max_abs_gi_n;
        for (size_t i = 0; i < gi_n.size(); ++i) {
            REQUIRE(std::abs(gi_n_reconst[i] - gi_n[i]) < tolerance);
        }
    }
    */
}

// Additional unit tests can be added similarly