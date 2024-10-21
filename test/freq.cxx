#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <iostream>

#include <sparseir/freq.hpp>

// Include the previously implemented classes and functions here

TEST_CASE("freq.jl", "[freq]") {
    SECTION("freq") {
        REQUIRE(zeta(MatsubaraFreq<Bosonic>(2)) == 0);
        REQUIRE(zeta(MatsubaraFreq<Fermionic>(-5)) == 1);

        REQUIRE(Int(FermionicFreq(3)) == 3);
        REQUIRE(Int(BosonicFreq(-2)) == -2);

        REQUIRE(Int(MatsubaraFreq<Bosonic>(4)) == 4);

        REQUIRE_THROWS_AS(FermionicFreq(4), std::domain_error);
        REQUIRE_THROWS_AS(BosonicFreq(-7), std::domain_error);

        REQUIRE(FermionicFreq(5) < BosonicFreq(6));
        REQUIRE(BosonicFreq(6) >= BosonicFreq(6));

        REQUIRE(value(pioverbeta, 3) == Approx(M_PI / 3));
        REQUIRE(valueim(2 * pioverbeta, 3) == std::complex<double>(0, 2 * M_PI / 3));

        REQUIRE(!MatsubaraFreq<Fermionic>(-3).isZero());
    }

    SECTION("freqadd") {
        REQUIRE(+pioverbeta == pioverbeta);
        REQUIRE((pioverbeta - pioverbeta).isZero());

        REQUIRE(pioverbeta + pioverbeta == 2 * pioverbeta);
        REQUIRE(Int(4 * pioverbeta) == 4);
        REQUIRE(Int(pioverbeta - 2 * pioverbeta) == -1);
        REQUIRE(zero(2 * pioverbeta).isZero());
    }

    SECTION("freqrange") {
        REQUIRE(std::distance(FermionicFreq(1), FermionicFreq(-3)) == 0);
        REQUIRE(std::distance(FermionicFreq(3), FermionicFreq(200000000001)) == 100000000000);

        REQUIRE(std::vector<BosonicFreq>(BosonicFreq(2), BosonicFreq(-2)).empty());
        REQUIRE(std::vector<BosonicFreq>(BosonicFreq(0), BosonicFreq(4)) == std::vector<BosonicFreq>{0, 2, 4} * pioverbeta);
        REQUIRE(std::vector<FermionicFreq>(FermionicFreq(-3), FermionicFreq(1)) == std::vector<FermionicFreq>{-3, -1, 1} * pioverbeta);

        REQUIRE(std::distance(FermionicFreq(37), BosonicFreq(10), FermionicFreq(87)) == 6);
        REQUIRE(std::vector<BosonicFreq>(BosonicFreq(-10), BosonicFreq(4), BosonicFreq(58)) == std::vector<BosonicFreq>{-10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58} * pioverbeta);
        REQUIRE(std::vector<BosonicFreq>(BosonicFreq(-10), BosonicFreq(4), BosonicFreq(60)) == std::vector<BosonicFreq>{-10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58} * pioverbeta);
        REQUIRE(std::distance(BosonicFreq(-10), BosonicFreq(4), BosonicFreq(60)) == 18);
        REQUIRE(std::distance(FermionicFreq(1), BosonicFreq(100), FermionicFreq(3)) == 1);
        REQUIRE(std::distance(FermionicFreq(1), BosonicFreq(100), FermionicFreq(-1001)) == 0);

        REQUIRE_THROWS_AS(std::vector<FermionicFreq>(FermionicFreq(5), BosonicFreq(100)), std::logic_error);
        REQUIRE_THROWS_AS(std::vector<BosonicFreq>(BosonicFreq(6), FermionicFreq(3), BosonicFreq(100)), std::logic_error);
        REQUIRE_THROWS_AS(std::vector<FermionicFreq>(FermionicFreq(7), FermionicFreq(3), FermionicFreq(101)), std::logic_error);
    }

    SECTION("freqinvalid") {
        REQUIRE_THROWS_AS(BosonicFreq(2) == 2, std::invalid_argument);
        REQUIRE_THROWS_AS(FermionicFreq(1) - 1, std::invalid_argument);
    }

    SECTION("unit tests") {
        REQUIRE_THROWS_AS(createStatistics(2), std::domain_error);
        REQUIRE(*createStatistics(0) + *createStatistics(1) == Fermionic());
        REQUIRE(Integer(FermionicFreq(19)) == 19);
        REQUIRE(-BosonicFreq(-24) == BosonicFreq(24));
        REQUIRE(sign(BosonicFreq(24)) == 1);
        REQUIRE(sign(BosonicFreq(0)) == 0);
        REQUIRE(sign(BosonicFreq(-94)) == -1);
        REQUIRE(BosonicFreq(24) % FermionicFreq(-7) == FermionicFreq(3));
        REQUIRE(FermionicFreq(123) % FermionicFreq(9) == BosonicFreq(6));
        REQUIRE(std::is_same<decltype(BosonicFreq), decltype(FermionicFreq)>::value);
    }
}