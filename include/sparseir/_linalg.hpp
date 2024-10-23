#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Vector;

template <typename T>
struct SVDResult {
    Matrix<T, Dynamic, Dynamic> U;
    Vector<T, Dynamic> s;
    Matrix<T, Dynamic, Dynamic> V;
};

template <typename T>
struct QRPivoted {
    Matrix<T, Dynamic, Dynamic> factors;
    Vector<T, Dynamic> taus;
    Vector<int, Dynamic> jpvt;
};

template <typename T>
QRPivoted<T> rrqr(Matrix<T, Dynamic, Dynamic>& A, double rtol = std::numeric_limits<double>::epsilon()) {
    int m = A.rows();
    int n = A.cols();
    int k = std::min(m, n);

    Vector<int, Dynamic> jpvt = Vector<int, Dynamic>::LinSpaced(n, 0, n - 1);
    Vector<T, Dynamic> taus(k);

    Vector<T, Dynamic> xnorms = A.colwise().norm();
    Vector<T, Dynamic> pnorms = xnorms;
    double sqrteps = std::sqrt(std::numeric_limits<double>::epsilon());

    for (int i = 0; i < k; ++i) {
        int pvt = (pnorms.segment(i, n - i).array().abs()).maxCoeff(&pvt) + i;
        if (i != pvt) {
            std::swap(jpvt[i], jpvt[pvt]);
            std::swap(xnorms[pvt], xnorms[i]);
            std::swap(pnorms[pvt], pnorms[i]);
            A.col(i).swap(A.col(pvt));
        }

        Eigen::HouseholderQR<Eigen::Ref<Matrix<T, Dynamic, Dynamic>>> qr(A.bottomRightCorner(m - i, n - i));
        taus[i] = qr.householderQ().coeff(0, 0);
        qr.applyHouseholderOnTheLeft(A.bottomRightCorner(m - i, n - i), qr.householderQ().col(0));

        for (int j = i + 1; j < n; ++j) {
            double temp = std::abs(A(i, j)) / pnorms[j];
            temp = std::max(0.0, (1.0 + temp) * (1.0 - temp));
            double temp2 = temp * std::pow(pnorms[j] / xnorms[j], 2);
            if (temp2 < sqrteps) {
                pnorms[j] = A.col(j).tail(m - i - 1).norm();
                xnorms[j] = pnorms[j];
            } else {
                pnorms[j] *= std::sqrt(temp);
            }
        }

        if (std::abs(A(i, i)) < rtol * std::abs(A(0, 0))) {
            A.bottomRightCorner(m - i, n - i).setZero();
            taus.tail(k - i).setZero();
            k = i;
            break;
        }
    }

    return {A, taus, jpvt};
}

template <typename T>
std::pair<Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic>> truncate_qr_result(const QRPivoted<T>& qr, int k) {
    int m = qr.factors.rows();
    int n = qr.factors.cols();
    if (k < 0 || k > std::min(m, n)) {
        throw std::domain_error("Invalid rank, must be in [0, " + std::to_string(std::min(m, n)) + "]");
    }

    Matrix<T, Dynamic, Dynamic> Q = qr.factors.leftCols(k).householderQr().householderQ();
    Matrix<T, Dynamic, Dynamic> R = qr.factors.topLeftCorner(k, n).triangularView<Eigen::Upper>();

    return {Q, R};
}

template <typename T>
SVDResult<T> tsvd(Matrix<T, Dynamic, Dynamic>& A, double rtol = std::numeric_limits<double>::epsilon()) {
    auto [A_qr, k] = rrqr(A, rtol);
    auto [Q, R] = truncate_qr_result(A_qr, k);

    Eigen::JacobiSVD<Matrix<T, Dynamic, Dynamic>> svd(R.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    Matrix<T, Dynamic, Dynamic> U = Q * svd.matrixV();
    Matrix<T, Dynamic, Dynamic> V = svd.matrixU().transpose();
    Vector<T, Dynamic> s = svd.singularValues();

    return {U, s, V};
}

std::tuple<double, double, double, double> svd2x2(double a11, double a12, double a21, double a22) {
    double abs_a12 = std::abs(a12);
    double abs_a21 = std::abs(a21);

    if (a21 == 0) {
        return svd2x2(a11, a12, a22);
    } else if (abs_a12 < abs_a21) {
        auto [cv, sv, smax, smin, cu, su] = svd2x2(a11, a21, a12, a22);
        return {cu, su, smax, smin, cv, sv};
    } else {
        auto [cx, sx, rx] = givens_params(a11, a21);
        a11 = rx;
        a21 = 0;
        std::tie(a12, a22) = givens_lmul(cx, sx, a12, a22);

        auto [cu, su, smax, smin, cv, sv] = svd2x2(a11, a12, a22);
        std::tie(cu, su) = givens_lmul(cx, -sx, cu, su);

        return {cu, su, smax, smin, cv, sv};
    }
}

std::tuple<double, double, double> givens_params(double f, double g) {
    if (g == 0) {
        return {1.0, 0.0, f};
    } else if (f == 0) {
        return {0.0, std::copysign(1.0, g), std::abs(g)};
    } else {
        double r = std::copysign(std::hypot(f, g), f);
        double c = f / r;
        double s = g / r;
        return {c, s, r};
    }
}

std::pair<double, double> givens_lmul(double c, double s, double x, double y) {
    double a = c * x + s * y;
    double b = c * y - s * x;
    return {a, b};
}

template <typename T>
double jacobi_sweep(Matrix<T, Dynamic, Dynamic>& U, Matrix<T, Dynamic, Dynamic>& VT) {
    int ii = U.rows();
    int jj = U.cols();
    if (ii < jj) {
        throw std::invalid_argument("matrix must be 'tall'");
    }
    if (VT.rows() != jj) {
        throw std::invalid_argument("U and VT must be compatible");
    }

    double offd = 0.0;
    for (int i = 0; i < ii; ++i) {
        for (int j = i + 1; j < jj; ++j) {
            double Hii = U.col(i).squaredNorm();
            double Hij = U.col(i).dot(U.col(j));
            double Hjj = U.col(j).squaredNorm();
            offd += Hij * Hij;

            auto [cv, sv, _, _] = svd2x2(Hii, Hij, Hij, Hjj);

            Eigen::JacobiRotation<double> rot(cv, sv);
            VT.applyOnTheLeft(j, i, rot);
            U.applyOnTheRight(i, j, rot.transpose());
        }
    }
    return std::sqrt(offd);
}

template <typename T>
SVDResult<T> svd_jacobi(Matrix<T, Dynamic, Dynamic>& U, double rtol = std::numeric_limits<double>::epsilon(), int maxiter = 20) {
    int m = U.rows();
    int n = U.cols();
    if (m < n) {
        throw std::invalid_argument("matrix must be 'tall'");
    }

    Matrix<T, Dynamic, Dynamic> VT = Matrix<T, Dynamic, Dynamic>::Identity(n, n);
    double Unorm = U.topLeftCorner(n, n).norm();

    for (int iter = 0; iter < maxiter; ++iter) {
        double offd = jacobi_sweep(U, VT);
        if (offd < rtol * Unorm) {
            break;
        }
    }

    Vector<T, Dynamic> s = U.colwise().norm();
    U.array().rowwise() /= s.transpose().array();

    return {U, s, VT};
}