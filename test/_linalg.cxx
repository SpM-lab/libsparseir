#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include "sparseir/_linalg.hpp"
#include "xprec/ddouble.hpp"

using namespace Eigen;
using namespace xprec;

TEST_CASE("Jacobi SVD", "[linalg]") {
    for (auto T : {DDouble, Float64x2}) {
        Matrix<T, Dynamic, Dynamic> A = Matrix<T, Dynamic, Dynamic>::Random(20, 10);
        auto [U, S, V] = svd_jacobi(A);
        Matrix<T, Dynamic, Dynamic> S_diag = S.asDiagonal();
        REQUIRE((U * S_diag * V.transpose()).isApprox(A));
    }
}

TEST_CASE("RRQR", "[linalg]") {
    for (auto T : {DDouble, Float64x2}) {
        Matrix<T, Dynamic, Dynamic> A = Matrix<T, Dynamic, Dynamic>::Random(40, 30);
        double A_eps = A.norm() * std::numeric_limits<T>::epsilon();
        auto [A_qr, A_rank] = rrqr(A);
        Matrix<T, Dynamic, Dynamic> A_rec = A_qr.Q * A_qr.R * A_qr.P.transpose();
        REQUIRE(A_rec.isApprox(A, 4 * A_eps));
        REQUIRE(A_rank == 30);
    }
}

TEST_CASE("RRQR Trunc", "[linalg]") {
    for (auto T : {DDouble, Float64x2}) {
        Vector<T, Dynamic> x = Vector<T, Dynamic>::LinSpaced(101, -1, 1);
        Matrix<T, Dynamic, Dynamic> A = x.array().pow(Vector<T, Dynamic>::LinSpaced(21, 0, 20).transpose().array());
        int m = A.rows();
        int n = A.cols();
        auto [A_qr, k] = rrqr(A, 1e-5);
        REQUIRE(k < std::min(m, n));

        auto [Q, R] = truncate_qr_result(A_qr, k);
        Matrix<T, Dynamic, Dynamic> A_rec = Q * R * A_qr.P.transpose();
        REQUIRE(A_rec.isApprox(A, 1e-5 * A.norm()));
    }
}

TEST_CASE("TSVD", "[linalg]") {
    for (auto T : {DDouble, Float64x2}) {
        for (auto tol : {1e-14, 1e-13}) {
            Vector<T, Dynamic> x = Vector<T, Dynamic>::LinSpaced(201, -1, 1);
            Matrix<T, Dynamic, Dynamic> A = x.array().pow(Vector<T, Dynamic>::LinSpaced(51, 0, 50).transpose().array());
            auto [U, S, V] = tsvd(A, tol);
            int k = S.size();

            Matrix<T, Dynamic, Dynamic> S_diag = S.asDiagonal();
            REQUIRE((U * S_diag * V.transpose()).isApprox(A, tol * A.norm()));
            REQUIRE((U.transpose() * U).isIdentity());
            REQUIRE((V.transpose() * V).isIdentity());
            REQUIRE(std::is_sorted(S.data(), S.data() + S.size(), std::greater<T>()));
            REQUIRE(k < std::min(A.rows(), A.cols()));

            Eigen::JacobiSVD<Matrix<double, Dynamic, Dynamic>> svd(A.cast<double>());
            REQUIRE(S.isApprox(svd.singularValues().head(k).cast<T>()));
        }
    }
}

TEST_CASE("SVD of VERY triangular 2x2", "[linalg]") {
    for (auto T : {DDouble, Float64x2}) {
        auto [cu, su, smax, smin, cv, sv] = svd2x2(T(1), T(1e100), T(1));
        REQUIRE(cu == Approx(1.0));
        REQUIRE(su == Approx(1e-100));
        REQUIRE(smax == Approx(1e100));
        REQUIRE(smin == Approx(1e-100));
        REQUIRE(cv == Approx(1e-100));
        REQUIRE(sv == Approx(1.0));
        Matrix<T, 2, 2> U, S, Vt, A;
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << T(1), T(1e100), T(0), T(1);
        REQUIRE((U * S * Vt).isApprox(A));

        std::tie(cu, su, smax, smin, cv, sv) = svd2x2(T(1), T(1e100), T(1e100));
        REQUIRE(cu == Approx(1 / std::sqrt(2)));
        REQUIRE(su == Approx(1 / std::sqrt(2)));
        REQUIRE(smax == Approx(std::sqrt(2) * 1e100));
        REQUIRE(smin == Approx(1 / std::sqrt(2)));
        REQUIRE(cv == Approx(5e-101));
        REQUIRE(sv == Approx(1.0));
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << T(1), T(1e100), T(0), T(1e100);
        REQUIRE((U * S * Vt).isApprox(A));

        std::tie(cu, su, smax, smin, cv, sv) = svd2x2(T(1e100), T(1e200), T(2));
        REQUIRE(cu == Approx(1.0));
        REQUIRE(su == Approx(2e-200));
        REQUIRE(smax == Approx(1e200));
        REQUIRE(smin == Approx(2e-100));
        REQUIRE(cv == Approx(1e-100));
        REQUIRE(sv == Approx(1.0));
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << T(1e100), T(1e200), T(0), T(2);
        REQUIRE((U * S * Vt).isApprox(A));

        std::tie(cu, su, smax, smin, cv, sv) = svd2x2(T(1e-100), T(1), T(1e-100));
        REQUIRE(cu == Approx(1.0));
        REQUIRE(su == Approx(1e-100));
        REQUIRE(smax == Approx(1.0));
        REQUIRE(smin == Approx(1e-200));
        REQUIRE(cv == Approx(1e-100));
        REQUIRE(sv == Approx(1.0));
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << T(1e-100), T(1), T(0), T(1e-100);
        REQUIRE((U * S * Vt).isApprox(A));
    }
}

TEST_CASE("SVD of 'more lower' 2x2", "[linalg]") {
    for (auto T : {DDouble, Float64x2}) {
        auto [cu, su, smax, smin, cv, sv] = svd2x2(T(1), T(1e-100), T(1e100), T(1));
        REQUIRE(cu == Approx(1e-100));
        REQUIRE(su == Approx(1.0));
        REQUIRE(smax == Approx(1e100));
        REQUIRE(std::abs(smin) < 1e-100); // should be ≈ 0.0, but x ≈ 0 is equivalent to x == 0
        REQUIRE(cv == Approx(1.0));
        REQUIRE(sv == Approx(1e-100));
        Matrix<T, 2, 2> U, S, Vt, A;
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << T(1), T(1e-100), T(1e100), T(1);
        REQUIRE((U * S * Vt).isApprox(A));
    }
}

TEST_CASE("Givens rotation of 2D vector - special cases", "[linalg]") {
    for (auto T : {DDouble, Float64x2}) {
        for (auto v : {std::vector<T>{42, 0}, std::vector<T>{-42, 0}, std::vector<T>{0, 42}, std::vector<T>{0, -42}, std::vector<T>{0, 0}}) {
            auto [c, s, r] = givens_params(v[0], v[1]);
            Matrix<T, 2, 2> R;
            R << c, s, -s, c;
            Vector<T, 2> Rv;
            Rv << r, T(0);
            Vector<T, 2> v_vec = Eigen::Map<Vector<T, 2>>(v.data());
            REQUIRE((R * v_vec).isApprox(Rv));
        }
    }
}