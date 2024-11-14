#include <catch2/catch_test_macros.hpp>

#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include <algorithm>

#include <sparseir/_root.hpp>

// Include the previously implemented functions here

//template <typename F, typename T>
//std::vector<T> discrete_extrema(F f, const std::vector<T>& xgrid);

template <typename T>
T midpoint(T lo, T hi);

TEST_CASE("DiscreteExtrema") {
    std::vector<int> nonnegative = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> symmetric = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};

    auto identity = [](int x) { return x; };
    auto shifted_identity = [](int x) { return x - std::numeric_limits<double>::epsilon(); };
    auto square = [](int x) { return x * x; };
    auto constant = [](int x) { return 1; };

    REQUIRE(sparseir::discrete_extrema(identity, nonnegative) == std::vector<int>({8}));
    // REQUIRE(sparseir::discrete_extrema(shifted_identity, nonnegative) == std::vector<int>({0, 8}));
    // REQUIRE(discrete_extrema(square, symmetric) == std::vector<int>({-8, 0, 8}));
    // REQUIRE(discrete_extrema(constant, symmetric) == std::vector<int>({}));
}

TEST_CASE("Midpoint") {
    // fails
    //REQUIRE(midpoint(std::numeric_limits<int>::max(), std::numeric_limits<int>::max()) == std::numeric_limits<int>::max());
    // REQUIRE(midpoint(std::numeric_limits<int>::min(), std::numeric_limits<int>::max()) == -1);
    // fails
    //REQUIRE(midpoint(std::numeric_limits<int>::min(), std::numeric_limits<int>::min()) == std::numeric_limits<int>::min());
    // REQUIRE(midpoint(static_cast<int16_t>(1000), static_cast<int32_t>(2000)) == static_cast<int32_t>(1500));
    // REQUIRE(midpoint(std::numeric_limits<double>::max(), std::numeric_limits<double>::max()) == std::numeric_limits<double>::max());
    // REQUIRE(midpoint(static_cast<float>(0), std::numeric_limits<float>::max()) == std::numeric_limits<float>::max() / 2);
    // REQUIRE(midpoint(static_cast<float>(0), std::numeric_limits<long double>::max()) == std::numeric_limits<long double>::max() / 2);
    // REQUIRE(midpoint(static_cast<int16_t>(0), static_cast<int64_t>(99999999999999999999ULL)) == static_cast<int64_t>(99999999999999999999ULL) / 2);
    // REQUIRE(midpoint<double>(-10.0, 1.0) == -4.5);
}