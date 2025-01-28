#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

using std::invalid_argument;

TEST_CASE("utils", "sortperm_rev")
{
    Eigen::VectorXd vec1(5);
    vec1 << 4, 3, 2, 1, 0;
    std::vector<size_t> sorted_indices1 = sparseir::sortperm_rev(vec1);
    REQUIRE(sorted_indices1.size() == vec1.size());
    std::vector<size_t> sorted_indices_expected1 = {0, 1, 2, 3, 4};
    REQUIRE(sorted_indices1 == sorted_indices_expected1);

    Eigen::VectorXd vec2(5);
    vec2 << 0, 1, 2, 3, 4;
    std::vector<size_t> sorted_indices2 = sparseir::sortperm_rev(vec2);
    REQUIRE(sorted_indices2.size() == vec2.size());
    std::vector<size_t> sorted_indices_expected2 = {4, 3, 2, 1, 0};
    REQUIRE(sorted_indices2 == sorted_indices_expected2);

    Eigen::VectorXd vec3(5);
    vec3 << 2, 1, 3, 0, 4;
    std::vector<size_t> sorted_indices3 = sparseir::sortperm_rev(vec3);
    REQUIRE(sorted_indices3.size() == vec3.size());
    std::vector<size_t> sorted_indices_expected3 = {4, 2, 0, 1, 3};
    REQUIRE(sorted_indices3 == sorted_indices_expected3);
}
