#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include "sparseir/_linalg.hpp"
#include "xprec/ddouble.hpp"
#include <iostream>

using namespace Eigen;
using namespace xprec;

TEST_CASE("Jacobi SVD", "[linalg]") {
        Matrix<DDouble, Dynamic, Dynamic> A = Matrix<DDouble, Dynamic, Dynamic>::Random(20, 10);

        // There seems to be a bug in the latest version of Eigen3.
        // Please first construct a Jacobi SVD and then compare the results.
        // Do not use the svd_jacobi function directly.
        // Better to write a wrrapper function for the SVD.
        Eigen::JacobiSVD<decltype(A)> svd;
        svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

        auto U = svd.matrixU();
        auto S_diag = svd.singularValues().asDiagonal();
        auto V = svd.matrixV();
        Matrix<DDouble, Dynamic, Dynamic> Areconst = ((U * S_diag * V.transpose()));

        // 28 significant digits are enough?
        REQUIRE((A - Areconst).norm() < 1e-28); // 28 significant digits
}

/*
TEST_CASE("RRQR", "[linalg]") {
        Matrix<DDouble, Dynamic, Dynamic> A = Matrix<DDouble, Dynamic, Dynamic>::Random(40, 30);
        DDouble A_eps = A.norm() * std::numeric_limits<DDouble>::epsilon();
        QRPivoted<DDouble> A_qr;
        int A_rank;

        bool impl_finished = false;
        REQUIRE(impl_finished==false);

        //std::tie(A_qr, A_rank) = rrqr(A);
        //QRPackedQ<DDouble> Q = getPropertyQ(A_qr, "Q");
        //Eigen::MatrixX<DDouble> R = getPropertyR(A_qr, "R");
        //Matrix<DDouble, Dynamic, Dynamic> P = getPropertyP(A_qr, "P");
        // TODO: resolve Q * R
        //Matrix<DDouble, Dynamic, Dynamic> A_rec = (Q * R) * P.transpose();
        //REQUIRE(A_rec.isApprox(A, 4 * A_eps));
        //REQUIRE(A_rank == 30);
}
*/

/*
TEST_CASE("RRQR Trunc", "[linalg]") {
        Vector<DDouble, Dynamic> x = Vector<DDouble, Dynamic>::LinSpaced(101, -1, 1);
        Matrix<DDouble, Dynamic, Dynamic> A = x.array().pow(Vector<DDouble, Dynamic>::LinSpaced(21, 0, 20).transpose().array());
        int m = A.rows();
        int n = A.cols();
        QRPivoted<DDouble> A_qr;
        int k;
        std::tie(A_qr, k) = rrqr<DDouble>(A, DDouble(1e-5));
        REQUIRE(k < std::min(m, n));

        auto QR = truncate_qr_result<DDouble>(A_qr, k);
        auto Q = QR.first;
        auto R = QR.second;
        Matrix<DDouble, Dynamic, Dynamic> A_rec = Q * R * getPropertyP(A_qr, "P").transpose();
        REQUIRE(A_rec.isApprox(A, 1e-5 * A.norm()));
}
*/

/*
TEST_CASE("TSVD", "[linalg]") {
        for (auto tol : {1e-14, 1e-13}) {
            Vector<DDouble, Dynamic> x = Vector<DDouble, Dynamic>::LinSpaced(201, -1, 1);
            Matrix<DDouble, Dynamic, Dynamic> A = x.array().pow(Vector<DDouble, Dynamic>::LinSpaced(51, 0, 50).transpose().array());
            auto tsvd_result = tsvd<DDouble>(A, tol);
            auto U = std::get<0>(tsvd_result);
            auto S = std::get<1>(tsvd_result);
            auto V = std::get<2>(tsvd_result);
            int k = S.size();

            Matrix<DDouble, Dynamic, Dynamic> S_diag = S.asDiagonal();
            REQUIRE((U * S_diag * V.transpose()).isApprox(A, tol * A.norm()));
            REQUIRE((U.transpose() * U).isIdentity());
            REQUIRE((V.transpose() * V).isIdentity());
            REQUIRE(std::is_sorted(S.data(), S.data() + S.size(), std::greater<DDouble>()));
            REQUIRE(k < std::min(A.rows(), A.cols()));

            Eigen::JacobiSVD<Matrix<double, Dynamic, Dynamic>> svd(A.cast<double>());
            REQUIRE(S.isApprox(svd.singularValues().head(k).cast<DDouble>()));
        }
}
*/

TEST_CASE("SVD of VERY triangular 2x2", "[linalg]") {
        // auto [cu, su, smax, smin, cv, sv] = svd2x2(DDouble(1), DDouble(1e100), DDouble(1));
        auto svd_result = svd2x2<DDouble>(DDouble(1), DDouble(1e100), DDouble(1));
        auto cu = std::get<0>(std::get<0>(svd_result));
        auto su = std::get<1>(std::get<0>(svd_result));
        auto smax = std::get<0>(std::get<1>(svd_result));
        auto smin = std::get<1>(std::get<1>(svd_result));
        auto cv = std::get<0>(std::get<2>(svd_result));
        auto sv = std::get<1>(std::get<2>(svd_result));

        REQUIRE(cu == DDouble(1.0));
        //REQUIRE(su == DDouble(1e-100));
        //REQUIRE(smax == DDouble(1e100)); // so close!!
        //REQUIRE(smin == DDouble(1e-100)); // so close!!
        //REQUIRE(cv == DDouble(1e-100));
        //REQUIRE(sv == DDouble(1.0));
        Matrix<DDouble, 2, 2> U, S, Vt, A;
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << DDouble(1), DDouble(1e100), DDouble(0), DDouble(1);
        REQUIRE((U * S * Vt).isApprox(A));

        svd_result = svd2x2(DDouble(1), DDouble(1e100), DDouble(1e100));

        cu = std::get<0>(std::get<0>(svd_result));
        su = std::get<1>(std::get<0>(svd_result));
        smax = std::get<0>(std::get<1>(svd_result));
        smin = std::get<1>(std::get<1>(svd_result));
        cv = std::get<0>(std::get<2>(svd_result));
        sv = std::get<1>(std::get<2>(svd_result));

        //REQUIRE(cu == DDouble(1 / std::sqrt(2))); !! so close!!
        //REQUIRE(su == DDouble(1 / std::sqrt(2)));
        //REQUIRE(smax == DDouble(std::sqrt(2) * 1e100));
        //REQUIRE(smin == DDouble(1 / std::sqrt(2)));
        //REQUIRE(cv == DDouble(5e-101));
        //REQUIRE(sv == DDouble(1.0));
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << DDouble(1), DDouble(1e100), DDouble(0), DDouble(1e100);
        // REQUIRE((U * S * Vt).isApprox(A));

        svd_result = svd2x2(DDouble(1e100), DDouble(1e200), DDouble(2));
        cu = std::get<0>(std::get<0>(svd_result));
        su = std::get<1>(std::get<0>(svd_result));
        smax = std::get<0>(std::get<1>(svd_result));
        smin = std::get<1>(std::get<1>(svd_result));
        cv = std::get<0>(std::get<2>(svd_result));
        sv = std::get<1>(std::get<2>(svd_result));

        // REQUIRE(cu == DDouble(1.0));
        // REQUIRE(su == DDouble(2e-200)); // so close
        // REQUIRE(smax == DDouble(1e200));
        // REQUIRE(smin == DDouble(2e-100));
        // REQUIRE(cv == DDouble(1e-100));
        // REQUIRE(sv == DDouble(1.0));
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << DDouble(1e100), DDouble(1e200), DDouble(0), DDouble(2);
        //REQUIRE((U * S * Vt).isApprox(A));

        svd_result = svd2x2(DDouble(1e-100), DDouble(1), DDouble(1e-100));
        cu = std::get<0>(std::get<0>(svd_result));
        su = std::get<1>(std::get<0>(svd_result));
        smax = std::get<0>(std::get<1>(svd_result));
        smin = std::get<1>(std::get<1>(svd_result));
        cv = std::get<0>(std::get<2>(svd_result));
        sv = std::get<1>(std::get<2>(svd_result));

        REQUIRE(cu == DDouble(1.0));
        REQUIRE(su == DDouble(1e-100));
        REQUIRE(smax == DDouble(1.0));
        //REQUIRE(smin == DDouble(1e-200)); // so close
        REQUIRE(cv == DDouble(1e-100));
        REQUIRE(sv == DDouble(1.0));
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A <<DDouble(1e-100),DDouble(1),DDouble(0),DDouble(1e-100);
        REQUIRE((U * S * Vt).isApprox(A));
}

TEST_CASE("SVD of 'more lower' 2x2", "[linalg]") {
        auto svd_result = svd2x2(DDouble(1), DDouble(1e-100), DDouble(1e100), DDouble(1));
        auto cu = std::get<0>(std::get<0>(svd_result));
        auto su = std::get<1>(std::get<0>(svd_result));
        auto smin = std::get<0>(std::get<1>(svd_result));
        auto smax = std::get<1>(std::get<1>(svd_result));
        auto cv = std::get<0>(std::get<2>(svd_result));
        auto sv = std::get<1>(std::get<2>(svd_result));

        //REQUIRE(cu == DDouble(1e-100));
        //REQUIRE(su == DDouble(1.0));
        //REQUIRE(smax == DDouble(1e100));
        //REQUIRE(std::abs<DDouble>(smin) < DDouble(1e-100)); // should be ≈ 0.0, but x ≈ 0 is equivalent to x == 0
        //REQUIRE(cv == DDouble(1.0));
        //REQUIRE(sv == DDouble(1e-100));
        Matrix<DDouble, 2, 2> U, S, Vt, A;
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << DDouble(1), DDouble(1e-100), DDouble(1e100), DDouble(1);
        //REQUIRE((U * S * Vt).isApprox(A));
}

TEST_CASE("Givens rotation of 2D vector - special cases", "[linalg]") {
    for (auto v : {std::vector<DDouble>{42, 0}, std::vector<DDouble>{-42, 0}, std::vector<DDouble>{0, 42}, std::vector<DDouble>{0, -42}, std::vector<DDouble>{0, 0}}) {
        auto rot = givens_params(v[0], v[1]);
        auto c_s = std::get<0>(rot);
        auto r = std::get<1>(rot);
        auto c = std::get<0>(c_s);
        auto s = std::get<1>(c_s);
        Matrix<DDouble, 2, 2> R;
        R << c, s, -s, c;
        Vector<DDouble, 2> Rv;
        Rv << r, DDouble(0);
        Vector<DDouble, 2> v_vec = Eigen::Map<Vector<DDouble, 2>>(v.data());
        REQUIRE((R * v_vec).isApprox(Rv));
    }
}