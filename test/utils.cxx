#include <unsupported/Eigen/CXX11/Tensor>

#include <catch2/catch_test_macros.hpp>

#include <sparseir/sparseir-header-only.hpp>
#include <xprec/ddouble-header-only.hpp>

using std::invalid_argument;

TEST_CASE("sortperm_rev", "[utils]")
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

TEST_CASE("issorted", "[utils]")
{
    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {5, 4, 3, 2, 1};
    Eigen::VectorXd ev1(3);
    ev1 << 1.0, 2.0, 3.0;

    REQUIRE(sparseir::issorted(v1) == true);
    REQUIRE(sparseir::issorted(v2) == false);
    REQUIRE(sparseir::issorted(ev1) == true);
}

    using ComplexF64 = std::complex<double>;

template<typename T, int N>
bool tensorIsApprox(const Eigen::Tensor<T, N>& a,
                    const Eigen::Tensor<T, N>& b,
                    double tol = 1e-12) {
    // Ensure dimensions match (assuming dimensions() returns an Eigen::DSizes<>)
    if (a.dimensions() != b.dimensions()) {
        return false;
    }
    for (int i = 0; i < a.size(); ++i) {
        // Compare the absolute difference of each complex number.
        if (std::abs(a.data()[i] - b.data()[i]) > tol) {
            return false;
        }
    }
    return true;
}


TEST_CASE("movedim", "[utils]"){
    const Eigen::Index s_size = 12;
    const Eigen::Index d1 = 2;
    const Eigen::Index d2 = 3;
    const Eigen::Index d3 = 4;

    Eigen::Tensor<ComplexF64, 1> s(s_size);
    Eigen::Tensor<ComplexF64, 4> rhol(s_size, d1, d2, d3);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    for (Eigen::Index i = 0; i < s_size; ++i) {
        s(i) = ComplexF64(dis(gen), dis(gen));
    }

    for (int i = 0; i < rhol.size(); ++i) {
        rhol.data()[i] = ComplexF64(dis(gen), dis(gen));
    }

    Eigen::array<Eigen::Index, 4> new_shape = {s_size, 1, 1, 1};
    Eigen::Tensor<ComplexF64, 4> s_reshaped = s.reshape(new_shape);
    Eigen::array<Eigen::Index, 4> bcast = {1, d1, d2, d3};
    Eigen::Tensor<ComplexF64, 4> originalgl = (-s_reshaped.broadcast(bcast)) * rhol;

    REQUIRE(originalgl.dimension(0) == s_size);
    REQUIRE(originalgl.dimension(1) == d1);
    REQUIRE(originalgl.dimension(2) == d2);
    REQUIRE(originalgl.dimension(3) == d3);

    Eigen::Tensor<ComplexF64, 4> moved_tensor1 = sparseir::movedim(originalgl, 0, 0);
    REQUIRE(tensorIsApprox(moved_tensor1, originalgl));

    REQUIRE(moved_tensor1.dimension(0) == s_size);
    REQUIRE(moved_tensor1.dimension(1) == d1);
    REQUIRE(moved_tensor1.dimension(2) == d2);
    REQUIRE(moved_tensor1.dimension(3) == d3);

    Eigen::Tensor<ComplexF64, 4> moved_tensor2 = sparseir::movedim(originalgl, 0, 1);

    REQUIRE(moved_tensor2.dimension(0) == d1);
    REQUIRE(moved_tensor2.dimension(1) == s_size);
    REQUIRE(moved_tensor2.dimension(2) == d2);
    REQUIRE(moved_tensor2.dimension(3) == d3);

    Eigen::Tensor<ComplexF64, 4> moved_tensor3 = sparseir::movedim(originalgl, 0, 2);

    REQUIRE(moved_tensor3.dimension(0) == d1);
    REQUIRE(moved_tensor3.dimension(1) == d2);
    REQUIRE(moved_tensor3.dimension(2) == s_size);
    REQUIRE(moved_tensor3.dimension(3) == d3);

    Eigen::Tensor<ComplexF64, 4> moved_tensor4 = sparseir::movedim(originalgl, 0, 3);

    REQUIRE(moved_tensor4.dimension(0) == d1);
    REQUIRE(moved_tensor4.dimension(1) == d2);
    REQUIRE(moved_tensor4.dimension(2) == d3);
    REQUIRE(moved_tensor4.dimension(3) == s_size);
}
