#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

#include <complex>
#include <memory>

using namespace sparseir;

TEST_CASE("MatsubaraPoles", "[dlr]")
{
    auto beta = 100.0;
    auto poles = std::vector<double>({1.0, 2.0, 3.0});
    auto e = 1e-12;
    auto n = 1;

    // Fermionic
    {
        auto polebasis =
            sparseir::MatsubaraPoles<sparseir::Fermionic>(beta, poles);
        auto result = polebasis.evaluate_frequency(1);
        for (auto i = 0; i < poles.size(); ++i) {
            auto w = std::complex<double>(0, 1) * M_PI * (2 * n + 1.0) / beta;
            REQUIRE(std::abs(result[i] - 1.0 / (w - poles[i])) < e);
        }
    }

    // Bosonic
    {
        auto polebasis =
            sparseir::MatsubaraPoles<sparseir::Bosonic>(beta, poles);
        //auto result = polebasis.evaluate_frequency(1);
        auto result = polebasis(n);
        for (auto i = 0; i < poles.size(); ++i) {
            auto w = std::complex<double>(0, 1) * M_PI * (2 * n + 0.0) / beta;
            REQUIRE(std::abs(result[i] - std::tanh(beta / 2 * poles[i]) /
                                             (w - poles[i])) < e);
        }
    }
}

TEST_CASE("TauPoles", "[dlr]")
{
    auto beta = 100.0;
    auto poles = std::vector<double>({1.0, 2.0, 3.0});
    auto e = 1e-12;
    // auto n = 1;

    // Fermionic
    {
        auto polebasis = sparseir::TauPoles<sparseir::Fermionic>(beta, poles);
        auto taus = std::vector<double>({0.0, 1.0, 2.0, 3.0, 4.0});
        auto results = polebasis(taus);
        for (auto p = 0; p < poles.size(); ++p) {
            for (auto t = 0; t < taus.size(); ++t) {
                auto ref = -std::exp(-poles[p] * taus[t]) /
                           (1.0 + std::exp(-beta * poles[p]));
                REQUIRE(std::abs(results(p, t) - ref) < e);
            }
        }
    }
    
    // TODO: Bosonic
}