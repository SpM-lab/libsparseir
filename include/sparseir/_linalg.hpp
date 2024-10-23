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
struct QRPackedQ {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> factors;
    Eigen::Matrix<T, Eigen::Dynamic, 1> taus;

    QRPackedQ(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& factors, const Eigen::Matrix<T, Eigen::Dynamic, 1>& taus)
        : factors(factors), taus(taus) {
        if (factors.rows() != taus.size()) {
            throw std::invalid_argument("The number of rows in factors must match the size of taus.");
        }
    }
};

template <typename T>
double reflector(Eigen::Vector<T, Dynamic>& x) {
    int n = x.size();
    if (n == 0) return 0.0;

    T xi1 = x(0);
    T normu = x.norm();
    if (normu == 0.0) return 0.0;

    T nu = std::copysign(normu, xi1);
    xi1 += nu;
    x(0) = -nu;

    for (int i = 1; i < n; ++i) {
        x(i) /= xi1;
    }

    return xi1 / nu;
}

template <typename T>
void reflectorApply(Eigen::Matrix<T, Eigen::Dynamic, 1>& x, T tau, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A) {
    int m = A.rows();
    int n = A.cols();
    if (x.size() != m) {
        throw std::invalid_argument("reflector has length " + std::to_string(x.size()) + ", which must match the first dimension of matrix A, " + std::to_string(m));
    }
    if (m == 0) return;

    for (int j = 0; j < n; ++j) {
        Eigen::Matrix<T, Eigen::Dynamic, 1> Aj = A.col(j).segment(1, m - 1);
        Eigen::Matrix<T, Eigen::Dynamic, 1> xj = x.segment(1, m - 1);
        T vAj = tau * (A(0, j) + xj.dot(Aj));
        A(0, j) -= vAj;
        Aj.noalias() -= vAj * xj;
    }
}

template <typename T>
std::pair<QRPivoted<T>, int> rrqr(Matrix<T, Dynamic, Dynamic>& A, double rtol = std::numeric_limits<double>::epsilon()) {
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

        T tau_i = reflector(A.col(i).tail(m - i));
        taus[i] = tau_i;
        refrectorApply(A.col(i).tail(m - i), tau_i, A.bottomRightCorner(m - i, n - i));

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

    return {QRPivoted<T>(A, taus, jpvt), k};
}

template <typename T>
void lmul(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& factors, const Eigen::Matrix<T, Eigen::Dynamic, 1>& taus, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& B) {
    int m = factors.rows();
    int n = factors.cols();
    int k = taus.size();

    if (B.rows() != m) {
        throw std::invalid_argument("The number of rows in B must match the number of rows in factors.");
    }

    // Apply the Householder reflections to B
    for (int i = 0; i < k; ++i) {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(m);
        v(i) = 1.0;
        v.tail(m - i - 1) = factors.col(i).tail(m - i - 1);
        B -= (taus(i) * v) * (v.transpose() * B);
    }
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& triu(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M, int k) {
    int m = M.rows();
    int n = M.cols();
    for (int j = 0; j < std::min(n, m + k); ++j) {
        for (int i = std::max(0, j - k + 1); i < m; ++i) {
            M(i, j) = 0;
        }
    }
    return M;
}

template <typename T>
std::pair<Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic>> truncate_qr_result(const QRPivoted<T>& qr, int k) {
    int m = qr.matrixQR().rows();
    int n = qr.matrixQR().cols();
    if (k < 0 || k > std::min(m, n)) {
        throw std::domain_error("Invalid rank, must be in [0, " + std::to_string(std::min(m, n)) + "]");
    }

    // Extract Q matrix
    auto Q = Eigen::Matrix<T, Dynamic, Dynamic>::Identity(m, k);
    lmul(qr.factors, qr.taus.head(k), Q);

    // Extract R matrix
    auto R = triu(qr.factors.topRows(k));

    return std::make_pair(Q, R);
}


template <typename T>
SVDResult<T> tsvd(Matrix<T, Dynamic, Dynamic>& A, double rtol = std::numeric_limits<double>::epsilon()) {
    auto A_qr_k = rrqr(A, rtol);
    auto QR_result = truncate_qr_result(A_qr_k.first, A_qr_k.second);
    auto Q = QR_result.first;
    auto R = QR_result.second;
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