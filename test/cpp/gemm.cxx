#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <complex>
#include <random>

#include <sparseir/sparseir.hpp> // C++ interface

using namespace Eigen;
using namespace sparseir;

// Helper function to create random matrices
template <typename T>
Matrix<T, Dynamic, Dynamic> create_random_matrix(int rows, int cols,
                                                 std::mt19937 &gen);

// Specialization for double
template <>
Matrix<double, Dynamic, Dynamic>
create_random_matrix<double>(int rows, int cols, std::mt19937 &gen)
{
    Matrix<double, Dynamic, Dynamic> mat(rows, cols);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (int i = 0; i < rows * cols; ++i) {
        mat.data()[i] = dist(gen);
    }
    return mat;
}

// Specialization for complex<double>
template <>
Matrix<std::complex<double>, Dynamic, Dynamic>
create_random_matrix<std::complex<double>>(int rows, int cols,
                                           std::mt19937 &gen)
{
    Matrix<std::complex<double>, Dynamic, Dynamic> mat(rows, cols);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (int i = 0; i < rows * cols; ++i) {
        mat.data()[i] = std::complex<double>(dist(gen), dist(gen));
    }
    return mat;
}

// Helper function to compare matrices with tolerance
template <typename T>
bool matrices_approximately_equal(const Matrix<T, Dynamic, Dynamic> &A,
                                  const Matrix<T, Dynamic, Dynamic> &B,
                                  double tolerance = 1e-10)
{
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            if (std::is_same<T, double>::value) {
                if (std::abs(A(i, j) - B(i, j)) > tolerance) {
                    return false;
                }
            } else if (std::is_same<T, std::complex<double>>::value) {
                if (std::abs(A(i, j) - B(i, j)) > tolerance) {
                    return false;
                }
            }
        }
    }
    return true;
}

TEST_CASE("_gemm_inplace basic functionality", "[contraction]")
{
    std::mt19937 gen(42); // Fixed seed for reproducible tests

    SECTION("double * double -> double")
    {
        int M = 3, N = 4, K = 2;

        Matrix<double, Dynamic, Dynamic> A =
            create_random_matrix<double>(M, K, gen);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(K, N, gen);
        Matrix<double, Dynamic, Dynamic> C_expected = A * B;
        Matrix<double, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        // Test _gemm_inplace
        _gemm_inplace<double, double, double>(A.data(), B.data(),
                                              C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("complex<double> * complex<double> -> complex<double>")
    {
        int M = 3, N = 4, K = 2;

        Matrix<std::complex<double>, Dynamic, Dynamic> A =
            create_random_matrix<std::complex<double>>(M, K, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> B =
            create_random_matrix<std::complex<double>>(K, N, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> C_expected = A * B;
        Matrix<std::complex<double>, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        // Test _gemm_inplace
        _gemm_inplace<std::complex<double>, std::complex<double>,
                      std::complex<double>>(A.data(), B.data(), C_actual.data(),
                                            M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("double * complex<double> -> complex<double>")
    {
        int M = 1, N = 1, K = 1;

        Matrix<double, Dynamic, Dynamic> A =
            create_random_matrix<double>(M, K, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> B =
            create_random_matrix<std::complex<double>>(K, N, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> C_expected =
            A.cast<std::complex<double>>() * B;
        Matrix<std::complex<double>, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        // Test _gemm_inplace
        _gemm_inplace<double, std::complex<double>, std::complex<double>>(
            A.data(), B.data(), C_actual.data(), M, N, K);

        // Debug output
        std::cout << "Expected C[0,0]: " << C_expected(0, 0) << std::endl;
        std::cout << "Actual C[0,0]: " << C_actual(0, 0) << std::endl;
        std::cout << "Expected C[0,1]: " << C_expected(0, 1) << std::endl;
        std::cout << "Actual C[0,1]: " << C_actual(0, 1) << std::endl;

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("complex<double> * double -> complex<double>")
    {
        int M = 3, N = 4, K = 2;

        Matrix<std::complex<double>, Dynamic, Dynamic> A =
            create_random_matrix<std::complex<double>>(M, K, gen);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(K, N, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> C_expected =
            A * B.cast<std::complex<double>>();
        Matrix<std::complex<double>, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        // Test _gemm_inplace
        _gemm_inplace<std::complex<double>, double, std::complex<double>>(
            A.data(), B.data(), C_actual.data(), M, N, K);

        // Debug output
        std::cout << "Expected C[0,0]: " << C_expected(0, 0) << std::endl;
        std::cout << "Actual C[0,0]: " << C_actual(0, 0) << std::endl;
        std::cout << "Expected C[0,1]: " << C_expected(0, 1) << std::endl;
        std::cout << "Actual C[0,1]: " << C_actual(0, 1) << std::endl;

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }
}

TEST_CASE("_gemm_inplace_t basic functionality", "[contraction]")
{
    std::mt19937 gen(42); // Fixed seed for reproducible tests

    SECTION("double * double -> double (B transposed)")
    {
        int M = 3, N = 4, K = 2;

        Matrix<double, Dynamic, Dynamic> A =
            create_random_matrix<double>(M, K, gen);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(N, K, gen); // Note: N rows, K cols
        Matrix<double, Dynamic, Dynamic> C_expected = A * B.transpose();
        Matrix<double, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        // Test _gemm_inplace_t
        _gemm_inplace_t<double, double, double>(A.data(), B.data(),
                                                C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION(
        "complex<double> * complex<double> -> complex<double> (B transposed)")
    {
        int M = 3, N = 4, K = 2;

        Matrix<std::complex<double>, Dynamic, Dynamic> A =
            create_random_matrix<std::complex<double>>(M, K, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> B =
            create_random_matrix<std::complex<double>>(N, K, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> C_expected =
            A * B.transpose();
        Matrix<std::complex<double>, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        // Test _gemm_inplace_t
        _gemm_inplace_t<std::complex<double>, std::complex<double>,
                        std::complex<double>>(A.data(), B.data(),
                                              C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("double * complex<double> -> complex<double> (B transposed)")
    {
        int M = 3, N = 4, K = 2;

        Matrix<double, Dynamic, Dynamic> A =
            create_random_matrix<double>(M, K, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> B =
            create_random_matrix<std::complex<double>>(N, K, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> C_expected =
            A.cast<std::complex<double>>() * B.transpose();
        Matrix<std::complex<double>, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        // Test _gemm_inplace_t
        _gemm_inplace_t<double, std::complex<double>, std::complex<double>>(
            A.data(), B.data(), C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("complex<double> * double -> complex<double> (B transposed)")
    {
        int M = 3, N = 4, K = 2;

        Matrix<std::complex<double>, Dynamic, Dynamic> A =
            create_random_matrix<std::complex<double>>(M, K, gen);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(N, K, gen);
        Matrix<std::complex<double>, Dynamic, Dynamic> C_expected =
            A * B.transpose().cast<std::complex<double>>();
        Matrix<std::complex<double>, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        // Test _gemm_inplace_t
        _gemm_inplace_t<std::complex<double>, double, std::complex<double>>(
            A.data(), B.data(), C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }
}

TEST_CASE("_gemm_inplace edge cases", "[contraction]")
{
    std::mt19937 gen(42);

    SECTION("1x1 matrices")
    {
        Matrix<double, Dynamic, Dynamic> A(1, 1);
        Matrix<double, Dynamic, Dynamic> B(1, 1);
        A(0, 0) = 2.5;
        B(0, 0) = 3.0;

        Matrix<double, Dynamic, Dynamic> C_expected = A * B;
        Matrix<double, Dynamic, Dynamic> C_actual(1, 1);
        C_actual.setZero();

        _gemm_inplace<double, double, double>(A.data(), B.data(),
                                              C_actual.data(), 1, 1, 1);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("Rectangular matrices")
    {
        int M = 5, N = 3, K = 7;

        Matrix<double, Dynamic, Dynamic> A =
            create_random_matrix<double>(M, K, gen);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(K, N, gen);
        Matrix<double, Dynamic, Dynamic> C_expected = A * B;
        Matrix<double, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        _gemm_inplace<double, double, double>(A.data(), B.data(),
                                              C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("Large matrices")
    {
        int M = 50, N = 30, K = 40;

        Matrix<double, Dynamic, Dynamic> A =
            create_random_matrix<double>(M, K, gen);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(K, N, gen);
        Matrix<double, Dynamic, Dynamic> C_expected = A * B;
        Matrix<double, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        _gemm_inplace<double, double, double>(A.data(), B.data(),
                                              C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }
}

TEST_CASE("_gemm_inplace_t edge cases", "[contraction]")
{
    std::mt19937 gen(42);

    SECTION("1x1 matrices")
    {
        Matrix<double, Dynamic, Dynamic> A(1, 1);
        Matrix<double, Dynamic, Dynamic> B(1, 1);
        A(0, 0) = 2.5;
        B(0, 0) = 3.0;

        Matrix<double, Dynamic, Dynamic> C_expected = A * B.transpose();
        Matrix<double, Dynamic, Dynamic> C_actual(1, 1);
        C_actual.setZero();

        _gemm_inplace_t<double, double, double>(A.data(), B.data(),
                                                C_actual.data(), 1, 1, 1);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("Rectangular matrices")
    {
        int M = 5, N = 3, K = 7;

        Matrix<double, Dynamic, Dynamic> A =
            create_random_matrix<double>(M, K, gen);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(N, K, gen);
        Matrix<double, Dynamic, Dynamic> C_expected = A * B.transpose();
        Matrix<double, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        _gemm_inplace_t<double, double, double>(A.data(), B.data(),
                                                C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("Large matrices")
    {
        int M = 50, N = 30, K = 40;

        Matrix<double, Dynamic, Dynamic> A =
            create_random_matrix<double>(M, K, gen);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(N, K, gen);
        Matrix<double, Dynamic, Dynamic> C_expected = A * B.transpose();
        Matrix<double, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        _gemm_inplace_t<double, double, double>(A.data(), B.data(),
                                                C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }
}

TEST_CASE("_gemm_inplace numerical accuracy", "[contraction]")
{
    std::mt19937 gen(42);

    SECTION("Identity matrix multiplication")
    {
        int size = 4;
        Matrix<double, Dynamic, Dynamic> A =
            Matrix<double, Dynamic, Dynamic>::Identity(size, size);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(size, size, gen);
        Matrix<double, Dynamic, Dynamic> C_expected = A * B;
        Matrix<double, Dynamic, Dynamic> C_actual(size, size);
        C_actual.setZero();

        _gemm_inplace<double, double, double>(
            A.data(), B.data(), C_actual.data(), size, size, size);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("Zero matrix multiplication")
    {
        int M = 3, N = 4, K = 2;
        Matrix<double, Dynamic, Dynamic> A =
            Matrix<double, Dynamic, Dynamic>::Zero(M, K);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(K, N, gen);
        Matrix<double, Dynamic, Dynamic> C_expected = A * B;
        Matrix<double, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        _gemm_inplace<double, double, double>(A.data(), B.data(),
                                              C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }
}

TEST_CASE("_gemm_inplace_t numerical accuracy", "[contraction]")
{
    std::mt19937 gen(42);

    SECTION("Identity matrix multiplication with transpose")
    {
        int size = 4;
        Matrix<double, Dynamic, Dynamic> A =
            Matrix<double, Dynamic, Dynamic>::Identity(size, size);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(size, size, gen);
        Matrix<double, Dynamic, Dynamic> C_expected = A * B.transpose();
        Matrix<double, Dynamic, Dynamic> C_actual(size, size);
        C_actual.setZero();

        _gemm_inplace_t<double, double, double>(
            A.data(), B.data(), C_actual.data(), size, size, size);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("Zero matrix multiplication with transpose")
    {
        int M = 3, N = 4, K = 2;
        Matrix<double, Dynamic, Dynamic> A =
            Matrix<double, Dynamic, Dynamic>::Zero(M, K);
        Matrix<double, Dynamic, Dynamic> B =
            create_random_matrix<double>(N, K, gen);
        Matrix<double, Dynamic, Dynamic> C_expected = A * B.transpose();
        Matrix<double, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        _gemm_inplace_t<double, double, double>(A.data(), B.data(),
                                                C_actual.data(), M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }
}

TEST_CASE("_gemm_inplace complex number edge cases", "[contraction]")
{
    std::mt19937 gen(42);

    SECTION("Pure real complex numbers")
    {
        int M = 3, N = 4, K = 2;

        Matrix<std::complex<double>, Dynamic, Dynamic> A(M, K);
        Matrix<std::complex<double>, Dynamic, Dynamic> B(K, N);

        for (int i = 0; i < M * K; ++i) {
            A.data()[i] = std::complex<double>(gen() % 10, 0.0);
        }
        for (int i = 0; i < K * N; ++i) {
            B.data()[i] = std::complex<double>(gen() % 10, 0.0);
        }

        Matrix<std::complex<double>, Dynamic, Dynamic> C_expected = A * B;
        Matrix<std::complex<double>, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        _gemm_inplace<std::complex<double>, std::complex<double>,
                      std::complex<double>>(A.data(), B.data(), C_actual.data(),
                                            M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }

    SECTION("Pure imaginary complex numbers")
    {
        int M = 3, N = 4, K = 2;

        Matrix<std::complex<double>, Dynamic, Dynamic> A(M, K);
        Matrix<std::complex<double>, Dynamic, Dynamic> B(K, N);

        for (int i = 0; i < M * K; ++i) {
            A.data()[i] = std::complex<double>(0.0, gen() % 10);
        }
        for (int i = 0; i < K * N; ++i) {
            B.data()[i] = std::complex<double>(0.0, gen() % 10);
        }

        Matrix<std::complex<double>, Dynamic, Dynamic> C_expected = A * B;
        Matrix<std::complex<double>, Dynamic, Dynamic> C_actual(M, N);
        C_actual.setZero();

        _gemm_inplace<std::complex<double>, std::complex<double>,
                      std::complex<double>>(A.data(), B.data(), C_actual.data(),
                                            M, N, K);

        REQUIRE(matrices_approximately_equal(C_expected, C_actual));
    }
}
