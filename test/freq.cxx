#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>

using std::invalid_argument;

TEST_CASE("freq", "[freq]")
{
    REQUIRE(sparseir::create_statistics(0)->zeta() == 0); // Bosonic
    REQUIRE(sparseir::create_statistics(1)->zeta() == 1); // Fermionic
}

TEST_CASE("freq", "[Integer value of FermionicFreq and BosonicFreq]")
{
    sparseir::FermionicFreq f(3);
    sparseir::BosonicFreq b(-2);
    REQUIRE(f.get_n() == 3);
    REQUIRE(b.get_n() == -2);
}

TEST_CASE("freq", "[Integer value of MatsubaraFreq with explicit integer cast]")
{
    sparseir::BosonicFreq bf(4);
    REQUIRE(bf.get_n() == 4);
}

TEST_CASE("freq",
          "[Exceptions for invalid FermionicFreq and BosonicFreq values]")
{
    REQUIRE_THROWS_AS(sparseir::FermionicFreq(4), std::domain_error);
    REQUIRE_THROWS_AS(sparseir::BosonicFreq(-7), std::domain_error);
}

TEST_CASE("freq", "[Comparison operators for FermionicFreq and BosonicFreq]")
{
    sparseir::FermionicFreq f(5);
    sparseir::BosonicFreq b(6);

    REQUIRE(f.get_n() < b.get_n());
    REQUIRE(b.get_n() >= b.get_n());
}

TEST_CASE("freq", "[Value calculation for MatsubaraFreq]")
{
    double beta = 3.0;
    sparseir::FermionicFreq bf(1);
    double expected_value = M_PI / beta;
    REQUIRE(bf.value(beta) == expected_value);
}

TEST_CASE("freq", "[Imaginary value calculation for MatsubaraFreq]")
{
    double beta = 3.0;
    sparseir::BosonicFreq bf(2);
    std::complex<double> expected_valueim(0, 2 * M_PI / beta);
    // std::cout << bf.valueim(beta) << std::endl;
    REQUIRE(bf.valueim(beta) == expected_valueim);
}

TEST_CASE("freq", "[iszero check for MatsubaraFreq]")
{
    sparseir::FermionicFreq f(-3);
    REQUIRE(!is_zero(f));

    sparseir::BosonicFreq bf(0);
    REQUIRE(is_zero(bf));

    sparseir::BosonicFreq bf_nonzero(2);
    REQUIRE(!is_zero(bf_nonzero));
}
