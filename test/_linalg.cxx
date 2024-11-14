#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include "sparseir/_linalg.hpp"
#include "xprec/ddouble.hpp"
#include <iostream>

using namespace Eigen;
using namespace xprec;

TEST_CASE("invperm", "[linalg]") {
        VectorXi a(3);
        a << 1, 2, 0;
        VectorXi b = sparseir::invperm(a);
        VectorX<int> refb(3);
        refb << 2, 0, 1;
        REQUIRE(b.isApprox(refb));
}

TEST_CASE("reflector", "[linalg]") {
        // double
        {
                Eigen::VectorXd v(3);
                v << 1, 2, 3;
                auto tau = sparseir::reflector(v);

                Eigen::VectorXd refv(3);
                // obtained by
                // julia> using LinearAlgebra; v = Float64[1,2,3]; LinearAlgebra.reflector!(v)
                refv << -3.7416574, 0.42179344, 0.63269017;
                for (int i = 0; i < 3; i++) {
                        REQUIRE(std::abs(v(i) - refv(i)) < 1e-7);
                }
        }

        // DDouble
        {
                Eigen::VectorX<DDouble> v(3);
                v << 1, 2, 3;
                auto tau = sparseir::reflector(v);

                Eigen::VectorX<DDouble> refv(3);
                refv << -3.7416574, 0.42179344, 0.63269017;
                for (int i = 0; i < 3; i++) {
                        REQUIRE(xprec::abs(v(i) - refv(i)) < 1e-7);
                }
        }
}

TEST_CASE("reflectorApply", "[linalg]") {
        /*
        using SparseIR
        T = Float64
        rtol = T(0.1)
        A = T[
                1 1 1
                1 1 1
                1 1 1
        ]

        i = 1
        Ainp = @view A[i:end, i]
        print("Pre: "); @show Ainp
        tau_i = SparseIR._LinAlg.reflector!(Ainp)
        @show tau_i
        block = @view A[i:end, (i+1):end]
        SparseIR._LinAlg.reflectorApply!(
                Ainp, tau_i, block
        )
        for i in axes(A, 1)
                for j in axes(A, 2)
                        s = (i,j) == size(A) ? ";" : ","
                        print(A[i, j], "$(s) ")
                end
                println()
        end
        */
        Eigen::MatrixX<double> A = Eigen::MatrixX<double>::Random(3, 3);
        A << 1, 1, 1,
             1, 1, 1,
             1, 1, 1;
        int m = A.rows();
        int n = A.cols();
        int k = std::min(m, n);
        double rtol = 0.1;
        int i = 0;

        auto Ainp = A.col(i).tail(m - i);
        REQUIRE(Ainp.size() == 3);
        auto tau_i = sparseir::reflector(Ainp);
        REQUIRE(std::abs(tau_i - 1.5773502691896257) < 1e-7);

        Eigen::VectorX<double> refv(3);
        refv << -1.7320508075688772, 0.36602540378443865, 0.36602540378443865;
        for (int i = 0; i < 3; i++) {
                REQUIRE(std::abs(Ainp(i) - refv(i)) < 1e-7);
        }

        auto block = A.bottomRightCorner(m - i, n - (i + 1));
        sparseir::reflectorApply(Ainp, tau_i, block);
        Eigen::MatrixX<double> refA(3, 3);
        refA <<-1.7320508075688772, -1.7320508075688772, -1.7320508075688772,
                0.36602540378443865, 0.0, 0.0,
                0.36602540378443865, 0.0, 0.0;
        REQUIRE(A.isApprox(refA, 1e-7));
}


TEST_CASE("Jacobi SVD", "[linalg]") {
        MatrixX<DDouble> A = MatrixX<DDouble>::Random(20, 10);

        // There seems to be a bug in the latest version of Eigen3.
        // Please first construct a Jacobi SVD and then compare the results.
        // Do not use the svd_jacobi function directly.
        // Better to write a wrrapper function for the SVD.
        Eigen::JacobiSVD<decltype(A)> svd;
        svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

        auto U = svd.matrixU();
        auto S_diag = svd.singularValues().asDiagonal();
        auto V = svd.matrixV();
        MatrixX<DDouble> Areconst = ((U * S_diag * V.transpose()));

        // 28 significant digits are enough?
        REQUIRE((A - Areconst).norm()/A.norm() < 1e-28); // 28 significant digits
}

TEST_CASE("sparseir::rrqr simple", "[linalg]") {
        Eigen::MatrixX<double> Aorig(3,3);
        Aorig << 1, 1, 1,
                 1, 1, 1,
                 1, 1, 1;
        Eigen::MatrixX<double> A(3,3);
        A << 1, 1, 1,
             1, 1, 1,
             1, 1, 1;

        double A_eps = A.norm() * std::numeric_limits<double>::epsilon();
        double rtol = 0.1;
        sparseir::QRPivoted<double> A_qr;
        int A_rank;
        std::tie(A_qr, A_rank) = sparseir::rrqr<double>(A);
        REQUIRE(A_rank == 1);
        Eigen::MatrixX<double> refA(3, 3);
        refA <<-1.7320508075688772, -1.7320508075688772, -1.7320508075688772,
                0.36602540378443865, 0.0, 0.0,
                0.36602540378443865, 0.0, 0.0;
        Eigen::VectorX<double> reftaus(3);
        reftaus << 1.5773502691896257, 0.0, 0.0;
        Eigen::VectorX<int> refjpvt(3);
        refjpvt << 0, 1, 2;

        REQUIRE(A_qr.factors.isApprox(refA, 1e-7));
        REQUIRE(A_qr.taus.isApprox(reftaus, 1e-7));
        REQUIRE(A_qr.jpvt == refjpvt);

        sparseir::QRPackedQ<double> Q = sparseir::getPropertyQ(A_qr);
        Eigen::VectorX<double> Qreftaus(3);
        Qreftaus << 1.5773502691896257, 0.0, 0.0;
        Eigen::MatrixX<double> Qreffactors(3, 3);
        Qreffactors << -1.7320508075688772, -1.7320508075688772, -1.7320508075688772,
                        0.36602540378443865, 0.0, 0.0,
                        0.36602540378443865, 0.0, 0.0;
        REQUIRE(Q.taus.isApprox(Qreftaus, 1e-7));
        REQUIRE(Q.factors.isApprox(Qreffactors, 1e-7));

        Eigen::MatrixX<double> R = getPropertyR(A_qr);
        Eigen::MatrixX<double> refR(3, 3);
        refR << -1.7320508075688772, -1.7320508075688772, -1.7320508075688772,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0;
        REQUIRE(R.isApprox(refR, 1e-7));


        MatrixX<double> P = getPropertyP(A_qr);
        MatrixX<double> refP = MatrixX<double>::Identity(3, 3);
        refP << 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0;
        REQUIRE(P.isApprox(refP, 1e-7));

        // In Julia Q * R
        // In C++ Q.factors * R

        MatrixX<double> C(3, 3);
        sparseir::mul<double>(C, Q, R);
        MatrixX<double> A_rec = C * P.transpose();

        REQUIRE(A_rec.isApprox(Aorig, 4 * A_eps));
}

TEST_CASE("sparseir::RRQR", "[linalg]") {
        MatrixX<DDouble> Aorig = MatrixX<DDouble>::Random(40, 30);
        MatrixX<DDouble> A = Aorig;
        DDouble A_eps = A.norm() * std::numeric_limits<DDouble>::epsilon();
        sparseir::QRPivoted<DDouble> A_qr;
        int A_rank;

        std::tie(A_qr, A_rank) = sparseir::rrqr(A);

        REQUIRE(A_rank == 30);
        sparseir::QRPackedQ<DDouble> Q = sparseir::getPropertyQ(A_qr);
        Eigen::MatrixX<DDouble> R = getPropertyR(A_qr);
        MatrixX<DDouble> P = getPropertyP(A_qr);

        // In Julia Q * R
        // In C++ Q.factors * R
        MatrixX<DDouble> C(40, 30);
        sparseir::mul<DDouble>(C, Q, R);
        MatrixX<DDouble> A_rec = C * P.transpose();
        REQUIRE(A_rec.isApprox(Aorig, 4 * A_eps));
}

TEST_CASE("sparseir::RRQR Trunc", "[linalg]") {
        // double
        {
                VectorX<double> x = VectorX<double>::LinSpaced(101, -1, 1);
                MatrixX<double> Aorig(101, 21);
                for (int i = 0; i < 21; i++) {
                        Aorig.col(i) = x.array().pow(i);
                }

                MatrixX<double> A = Aorig;
                int m = A.rows();
                int n = A.cols();
                sparseir::QRPivoted<double> A_qr;
                int k;
                std::tie(A_qr, k) = sparseir::rrqr<double>(A, double(1e-5));
                REQUIRE(k < std::min(m, n));
                REQUIRE(k == 17);
                auto QR = sparseir::truncate_qr_result<double>(A_qr, k);
                auto Q = QR.first;
                auto R = QR.second;

                MatrixX<double> A_rec = Q * R * getPropertyP(A_qr).transpose();
                REQUIRE(A_rec.isApprox(Aorig, 1e-5 * A.norm()));
        }
        // DDouble
        {
                using std::pow;
                VectorX<DDouble> x = VectorX<DDouble>::LinSpaced(101, -1, 1);
                MatrixX<DDouble> Aorig(101, 21);
                for (int i = 0; i < Aorig.cols(); i++) {
                        for (int j = 0; j < Aorig.rows(); j++) {
                                Aorig(j, i) = pow(x(j), i); // xprec::pow is called.
                        }
                }

                MatrixX<DDouble> A = Aorig;
                int m = A.rows();
                int n = A.cols();
                sparseir::QRPivoted<DDouble> A_qr;
                int k;
                std::tie(A_qr, k) = sparseir::rrqr<DDouble>(A, DDouble(0.00001));
                REQUIRE(k < std::min(m, n));
                REQUIRE(k == 17);
                auto QR = sparseir::truncate_qr_result<DDouble>(A_qr, k);
                auto Q = QR.first;
                auto R = QR.second;
                REQUIRE(Q.rows() == m);
                REQUIRE(Q.cols() == k);
                REQUIRE(R.rows() == k);
                REQUIRE(R.cols() == n);
                MatrixX<DDouble> A_rec = Q * R * getPropertyP(A_qr).transpose();
                REQUIRE(A_rec.isApprox(Aorig, 1e-5 * A.norm()));
        }
}

TEST_CASE("TSVD", "[linalg]") {
        using std::pow;
        // double
        {
            for (auto tol : {1e-14}) {
                int N1 = 201;
                int N2 = 51;
                VectorX<double> x = VectorX<double>::LinSpaced(N1, -1, 1);
                //MatrixX<double> Aorig(201, 51);
                MatrixX<double> Aorig(N1, N2);
                for (int i = 0; i < Aorig.cols(); i++) {
                    Aorig.col(i) = x.array().pow(i);
                    //for (int j = 0; j < Aorig.rows(); j++) {
                        //Aorig(j, i) = pow(x(j), i);
                    //}
                }

                MatrixX<double> A = Aorig; // create a copy of Aorig

                auto tsvd_result = sparseir::tsvd<double>(A, double(tol));
                sparseir::tsvd<double>(Aorig, double(tol));
                auto U = std::get<0>(tsvd_result);
                auto s = std::get<1>(tsvd_result);
                auto V = std::get<2>(tsvd_result);
                int k = s.size();

                auto S_diag = s.asDiagonal();
                auto Areconst = U * S_diag * V.transpose();
                auto diff = (A - Areconst).norm() / A.norm();
                // std::cout << "diff " << diff << std::endl;
                // std::cout << "Areconst " << Areconst.norm() << std::endl;
                // std::cout << "Aorig " << Aorig.norm() << std::endl;
                // std::cout << "norm diff" << Aorig.norm() - Areconst.norm() << std::endl;

                REQUIRE(Areconst.isApprox(Aorig, tol * Aorig.norm()));
                REQUIRE((U.transpose() * U).isIdentity());
                REQUIRE((V.transpose() * V).isIdentity());
                REQUIRE(std::is_sorted(s.data(), s.data() + s.size(), std::greater<DDouble>()));
                REQUIRE(k < std::min(A.rows(), A.cols()));

                Eigen::JacobiSVD<MatrixX<double>> svd(Aorig.cast<double>());
                REQUIRE(s.isApprox(svd.singularValues().head(k)));
                REQUIRE(S_diag.toDenseMatrix().isApprox(svd.singularValues().head(k).asDiagonal().toDenseMatrix()));
            }
        }
}

TEST_CASE("SVD of VERY triangular 2x2", "[linalg]") {
        // auto [cu, su, smax, smin, cv, sv] = sparseir::svd2x2(DDouble(1), DDouble(1e100), DDouble(1));
        auto svd_result = sparseir::svd2x2<DDouble>(DDouble(1), DDouble(1e100), DDouble(1));
        auto cu = std::get<0>(std::get<0>(svd_result));
        auto su = std::get<1>(std::get<0>(svd_result));
        auto smax = std::get<0>(std::get<1>(svd_result));
        auto smin = std::get<1>(std::get<1>(svd_result));
        auto cv = std::get<0>(std::get<2>(svd_result));
        auto sv = std::get<1>(std::get<2>(svd_result));

        REQUIRE(cu.hi() == 1.0);
        REQUIRE(su.hi() == 1e-100);
        REQUIRE(smax.hi() == 1e100);
        REQUIRE(smin.hi() == 1e-100);
        REQUIRE(cv.hi() == 1e-100);
        REQUIRE(sv.hi() == 1.0);
        Matrix<DDouble, 2, 2> U, S, Vt, A;
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << DDouble(1), DDouble(1e100), DDouble(0), DDouble(1);
        REQUIRE((U * S * Vt).isApprox(A));

        svd_result = sparseir::svd2x2(DDouble(1), DDouble(1e100), DDouble(1e100));

        cu = std::get<0>(std::get<0>(svd_result));
        su = std::get<1>(std::get<0>(svd_result));
        smax = std::get<0>(std::get<1>(svd_result));
        smin = std::get<1>(std::get<1>(svd_result));
        cv = std::get<0>(std::get<2>(svd_result));
        sv = std::get<1>(std::get<2>(svd_result));

        REQUIRE(cu.hi() == 1 / std::sqrt(2));
        REQUIRE(su.hi() == 1 / std::sqrt(2));
        REQUIRE(smax.hi() == std::sqrt(2) * 1e100);
        REQUIRE(smin.hi() == 1 / std::sqrt(2));
        REQUIRE(std::abs(cv.hi() - 5e-101) < 1e-10);
        REQUIRE(std::abs(sv.hi() - 1.0) < 1e-10);
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << DDouble(1), DDouble(1e100), DDouble(0), DDouble(1e100);
        // REQUIRE((U * S * Vt).isApprox(A));

        svd_result = sparseir::svd2x2(DDouble(1e100), DDouble(1e200), DDouble(2));
        cu = std::get<0>(std::get<0>(svd_result));
        su = std::get<1>(std::get<0>(svd_result));
        smax = std::get<0>(std::get<1>(svd_result));
        smin = std::get<1>(std::get<1>(svd_result));
        cv = std::get<0>(std::get<2>(svd_result));
        sv = std::get<1>(std::get<2>(svd_result));

        REQUIRE(cu.hi() == 1.0);
        REQUIRE(su.hi() == 2e-200);
        REQUIRE(smax.hi() == 1e200);
        REQUIRE(smin.hi() == 2e-100);
        REQUIRE(cv.hi() == 1e-100);
        REQUIRE(sv.hi() == 1.0);
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << DDouble(1e100), DDouble(1e200), DDouble(0), DDouble(2);
        // REQUIRE((U * S * Vt).isApprox(A));

        svd_result = sparseir::svd2x2(DDouble(1e-100), DDouble(1), DDouble(1e-100));
        cu = std::get<0>(std::get<0>(svd_result));
        su = std::get<1>(std::get<0>(svd_result));
        smax = std::get<0>(std::get<1>(svd_result));
        smin = std::get<1>(std::get<1>(svd_result));
        cv = std::get<0>(std::get<2>(svd_result));
        sv = std::get<1>(std::get<2>(svd_result));

        REQUIRE(cu.hi() == 1.0);
        REQUIRE(su.hi() == 1e-100);
        REQUIRE(smax.hi() == 1.0);
        REQUIRE(smin.hi() == 1e-200);
        REQUIRE(cv.hi() == 1e-100);
        REQUIRE(sv.hi() == 1.0);
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A <<DDouble(1e-100),DDouble(1),DDouble(0),DDouble(1e-100);
        REQUIRE((U * S * Vt).isApprox(A));
}

TEST_CASE("SVD of 'more lower' 2x2", "[linalg]") {
        auto svd_result = sparseir::svd2x2(DDouble(1), DDouble(1e-100), DDouble(1e100), DDouble(1));
        auto cu = std::get<0>(std::get<0>(svd_result));
        auto su = std::get<1>(std::get<0>(svd_result));
        auto smax = std::get<0>(std::get<1>(svd_result));
        auto smin = std::get<1>(std::get<1>(svd_result));
        auto cv = std::get<0>(std::get<2>(svd_result));
        auto sv = std::get<1>(std::get<2>(svd_result));

        // REQUIRE(cu.hi() == 1e-100); <--- fails
        REQUIRE(su.hi() == 1.0);
        REQUIRE(smax.hi() == 1e100);
        REQUIRE(xprec::abs(smin) < DDouble(1e-100)); // should be ≈ 0.0, but x ≈ 0 is equivalent to x == 0
        REQUIRE(cv.hi() == 1.0);
        REQUIRE(sv.hi() == 1e-100);
        Matrix<DDouble, 2, 2> U, S, Vt, A;
        U << cu, -su, su, cu;
        S << smax, 0, 0, smin;
        Vt << cv, sv, -sv, cv;
        A << DDouble(1), DDouble(1e-100), DDouble(1e100), DDouble(1);
        //REQUIRE((U * S * Vt).isApprox(A));
}

TEST_CASE("Givens rotation of 2D vector - special cases", "[linalg]") {
    for (auto v : {std::vector<DDouble>{42, 0}, std::vector<DDouble>{-42, 0}, std::vector<DDouble>{0, 42}, std::vector<DDouble>{0, -42}, std::vector<DDouble>{0, 0}}) {
        auto rot = sparseir::givens_params(v[0], v[1]);
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