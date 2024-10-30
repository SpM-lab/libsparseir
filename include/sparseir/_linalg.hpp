#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

#include "xprec/ddouble.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixX;
using Eigen::Vector;
using Eigen::VectorX;

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

/*
function getproperty(F::QRPivoted{T}, d::Symbol) where T
    m, n = size(F)
    if d === :R
        return triu!(getfield(F, :factors)[1:min(m,n), 1:n])
    elseif d === :Q
        return QRPackedQ(getfield(F, :factors), F.τ)
    elseif d === :p
        return getfield(F, :jpvt)
    elseif d === :P
        p = F.p
        n = length(p)
        P = zeros(T, n, n)
        for i in 1:n
            P[p[i],i] = one(T)
        end
        return P
    else
        getfield(F, d)
    end
end
*/


/*
template <typename T>
struct QRPackedQ {
    Matrix<T, Dynamic, Dynamic> factors;
    Matrix<T, Dynamic, 1> taus;

    QRPackedQ(const Matrix<T, Dynamic, Dynamic>& factors, const Matrix<T, Dynamic, 1>& taus)
        : factors(factors), taus(taus) {}
};
*/

template <typename T>
Matrix<T, Dynamic, Dynamic> triu(const Eigen::Block<const Matrix<T, Dynamic, Dynamic>, -1, -1, false>& M) {
    Matrix<T, Dynamic, Dynamic> upper = M;
    for (int i = 0; i < upper.rows(); ++i) {
        for (int j = 0; j < i; ++j) {
            upper(i, j) = 0;
        }
    }
    return upper;
}

// TODO: FIX THIS
template <typename T>
auto getPropertyP(const QRPivoted<T>& F, const std::string& property) {
    int m = F.factors.rows();
    int n = F.factors.cols();

        Matrix<T, Dynamic, Dynamic> P = Matrix<T, Dynamic, Dynamic>::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            P(F.jpvt[i], i) = 1;
        }
        return P;
}

// TODO: FIX THIS
template <typename T>
auto getPropertyQ(const QRPivoted<T>& F, const std::string& property) {
    return QRPackedQ<T>(F.factors, F.taus);
}

// TODO: FIX THIS
template <typename T>
auto getPropertyR(const QRPivoted<T>& F, const std::string& property) {
    int m = F.factors.rows();
    int n = F.factors.cols();
    return triu(F.factors.topLeftCorner(std::min(m, n), n));
}

template <typename T>
T reflector(Eigen::VectorBlock<Eigen::Matrix<T, Eigen::Dynamic, 1>>& x) {
    int n = x.size();
    if (n == 0) return static_cast<T>(0.0);

    T xi1 = x(0);
    T normu = x.norm();
    if (normu == static_cast<T>(0.0)) return static_cast<T>(0.0);

    T nu = std::copysign(normu, xi1);
    xi1 += nu;
    x(0) = -nu;

    for (int i = 1; i < n; ++i) {
        x(i) /= xi1;
    }

    return xi1 / nu;
}

template <typename T>
int myargmax(const Eigen::VectorX<T>& v, int start = 0) {
    int index = start;
    T M = v(start);
    for (int i = start; i < v.size(); ++i) {
        if (v(i) > M) {
            M = v(i);
            index = i;
        }
    }
    return index;
}

// Eigen::VectorBlock<Eigen::Block<Eigen::Matrix<xprec::DDouble, -1, -1>, -1, 1, true>>, T = xprec::DDouble, T3 = Eigen::Block<Eigen::Matrix<xprec::DDouble, -1, -1>>
template <typename T>
void reflectorApply(Eigen::VectorBlock<Eigen::Block<Eigen::Matrix<T, -1, -1>, -1, 1, true>>& x, T tau, Eigen::Block<Eigen::Matrix<T, -1, -1>>& A) {
    int m = A.rows();
    int n = A.cols();

    if (x.size() != m) {
        throw std::invalid_argument("Reflector length must match the first dimension of matrix A");
    }

    if (m == 0) return;

    // Loop over each column of A
    for (int j = 0; j < n; ++j) {
        // Equivalent to `Aj = view(A, 2:m, j)` and `xj = view(x, 2:m)`
        Eigen::Block<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> Aj = A.block(1, j, m - 1, 1);
        Eigen::Block<Eigen::Matrix<T, Eigen::Dynamic, 1>> xj = x.tail(m - 1);

        // Compute vAj = conj(tau) * (A(0, j) + xj.dot(Aj));
        T vAj = std::conj(tau) * (A(0, j) + xj.dot(Aj));

        // Update A(0, j)
        A(0, j) -= vAj;

        // Apply axpy operation: Aj -= vAj * xj
        Aj.noalias() -= vAj * xj;
    }
}

template <typename T>
std::pair<QRPivoted<T>, int> rrqr(MatrixX<T>& A, T rtol = std::numeric_limits<T>::epsilon()) {
    int m = A.rows();
    int n = A.cols();
    int k = std::min(m, n);

    Vector<int, Dynamic> jpvt = Vector<int, Dynamic>::LinSpaced(n, 0, n - 1);
    Vector<T, Dynamic> taus(k);

    Vector<T, Dynamic> xnorms = A.colwise().norm();
    Vector<T, Dynamic> pnorms = xnorms;
    T sqrteps = T(std::sqrt(std::numeric_limits<double>::epsilon()));

    for (int i = 0; i < k; ++i) {
        int pvt = myargmax(pnorms, n-i) + i;
        if (i != pvt) {
            std::swap(jpvt[i], jpvt[pvt]);
            std::swap(xnorms[pvt], xnorms[i]);
            std::swap(pnorms[pvt], pnorms[i]);
            A.col(i).swap(A.col(pvt));
        }
        auto x_ = A.col(i).tail(m - i);

        // inline implementation
        // T tau_i = reflector(x);
        int n = x_.size();
        T tau_i;
        if (n == 0) {
            tau_i = static_cast<T>(0.0);
        } else {
            T xi1 = x_(0);
            T normu = x_.norm();
            if (normu == static_cast<T>(0.0)){
                tau_i = static_cast<T>(0.0);
            }
            // Fix me: `std::copysign<T>` is not available for T = DDouble
            T nu = T(std::copysign((double)normu, (double)(xi1)));
            xi1 += nu;
            x_(0) = -nu;

            for (int i = 1; i < n; ++i) {
                x_(i) /= xi1;
            }

            T tau_i = xi1 / nu;
        }
        // end inline implementation of reflector
        taus[i] = tau_i;
        // TODO: fix me
        MatrixX<T> Ainp = A.col(i).tail(m - i);
        reflectorApply(Ainp, tau_i, A.bottomRightCorner(m - i, n - i));

        for (int j = i + 1; j < n; ++j) {
            T temp = std::abs((double)(A(i, j))) / pnorms[j];
            temp = std::max<T>(0.0, (1.0 + temp) * (1.0 - temp));
            // abs2
            T temp2 = temp * (pnorms(j) / xnorms(j)) * (pnorms(j) / xnorms(j));
            if (temp2 < sqrteps) {
                pnorms(j) = A.col(j).tail(m - i - 1).norm();
                xnorms(j) = pnorms(j);
            } else {
                pnorms(j) = pnorms(j) * T(std::sqrt(double(temp)));
            }
        }

        if (std::abs((double)(A(i,i))) < rtol * std::abs((double)(A(0,0)))) {
            A.bottomRightCorner(m - i, n - i).setZero();
            taus.tail(k - i).setZero();
            k = i;
            break;
        }
    }

    return {QRPivoted<T>{A, taus, jpvt}, k};
}

/*
template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, std::vector<int>>
rrqr(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A, T rtol = std::numeric_limits<T>::epsilon()) {
    int m = A.rows();
    int n = A.cols();
    int k = std::min(m, n);

    // Initialize pivot vector with default ordering
    Vector<int, Dynamic> jpvt = Vector<int, Dynamic>::LinSpaced(n, 0, n - 1);

    // Norms of columns
    Eigen::VectorXd xnorms = A.colwise().norm();
    Eigen::VectorXd pnorms = xnorms;
    T sqrteps = std::sqrt(std::numeric_limits<T>::epsilon());

    // Begin QR factorization with pivoting
    for (int i = 0; i < k; ++i) {
        // Select the pivot column
        int pvt;
        pnorms.segment(i, n - i).maxCoeff(&pvt);
        pvt += i;

        if (i != pvt) {
            std::swap(jpvt[i], jpvt[pvt]);
            std::swap(xnorms[i], xnorms[pvt]);
            std::swap(pnorms[i], pnorms[pvt]);
            swapCols(A, i, pvt);
        }

        T tau = Eigen::internal::householder_qr_inplace(A.col(i).tail(m - i));
        A.col(i).tail(m - i) = -tau;

        for (int j = i + 1; j < n; ++j) {
            T temp = std::abs(A(i, j)) / pnorms[j];
            temp = std::max(T(0), (T(1) + temp) * (T(1) - temp));
            // abs2
            T temp2 = temp * (pnorms[j] / xnorms[j]) * (pnorms[j] / xnorms[j]);
            if (temp2 < sqrteps) {
                pnorms[j] = A.block(i + 1, j, m - i - 1, 1).norm();
                xnorms[j] = pnorms[j];
            } else {
                pnorms[j] *= std::sqrt(temp);
            }
        }

        if (std::abs(A(i, i)) < rtol * std::abs(A(0, 0))) {
            A.block(i, i, m - i, n - i).setZero();
            k = i;
            break;
        }
    }

    return {QRPivoted<T>{A, taus, jpvt}, k};
}
*/

template <typename T>
void lmul(const Eigen::Matrix<T, -1, -1>& factors, Eigen::VectorBlock<const Eigen::Matrix<T, -1, 1>> taus, Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<T>, Eigen::Matrix<T, -1, -1>> B){
//void lmul(T1& factors, const T2& taus, T3& B) {
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
        // Fix me
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
    int m = qr.factors.rows();
    int n = qr.factors.cols();
    if (k < 0 || k > std::min(m, n)) {
        throw std::domain_error("Invalid rank, must be in [0, " + std::to_string(std::min(m, n)) + "]");
    }

    // Extract Q matrix
    auto Q = Eigen::Matrix<T, Dynamic, Dynamic>::Identity(m, k);
    lmul<>(qr.factors, qr.taus.head(k), Q);

    // Extract R matrix
    auto R = triu(qr.factors.topRows(k));

    return std::make_pair(Q, R);
}

/*
template <typename T>
std::pair<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
truncate_qr_result(const Eigen::HouseholderQR<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& qr, int k) {
    int m = qr.matrixQR().rows();
    int n = qr.matrixQR().cols();

    // Ensure k is within the valid range [0, min(m, n)]
    if (!(0 <= k && k <= std::min(m, n))) {
        throw std::domain_error("Invalid rank, must be in [0, min(m, n)]");
    }

    // Extract the first k columns of Q
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Qfull = qr.householderQ() * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(m, k);

    // Extract the upper triangular part of the first k rows of R
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R = qr.matrixQR().topLeftCorner(k, n).template triangularView<Eigen::Upper>();

    return std::make_pair(Qfull, R);
}
*/


/*
template <typename T>
SVDResult<T> tsvd(Matrix<T, Dynamic, Dynamic>& A, double rtol = std::numeric_limits<double>::epsilon()) {
    auto A_qr_k = rrqr<T>(A, rtol);
    // FIX ME
    auto QR_result = truncate_qr_result<T>(A_qr_k.first, A_qr_k.second);
    auto Q = QR_result.first;
    auto R = QR_result.second;
    Eigen::JacobiSVD<Matrix<T, Dynamic, Dynamic>> svd(R.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    Matrix<T, Dynamic, Dynamic> U = Q * svd.matrixV();
    Matrix<T, Dynamic, Dynamic> V = svd.matrixU().transpose();
    Vector<T, Dynamic> s = svd.singularValues();

    return SVDResult<T> {U, s, V};
}
*/


// Swap columns of a matrix A
template <typename T>
void swapCols(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A, int i, int j) {
    if (i != j) {
        A.col(i).swap(A.col(j));
    }
}

// Truncate RRQR result to low rank
template <typename T>
std::pair<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
truncateQRResult(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Q, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& R, int k) {
    int m = Q.rows();
    int n = R.cols();

    if (k < 0 || k > std::min(m, n)) {
        throw std::domain_error("Invalid rank, must be in [0, min(m, n)]");
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Q_trunc = Q.leftCols(k);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> R_trunc = R.topLeftCorner(k, n);
    return std::make_pair(Q_trunc, R_trunc);
}

// Truncated SVD (TSVD)
template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::VectorXd, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
tsvd(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& A, T rtol = std::numeric_limits<T>::epsilon()) {
    // Step 1: Apply RRQR to A
    std::pair<QRPivoted<T>, int>
    rrqr_result = rrqr(A, rtol);
    auto A_qr = std::get<0>(rrqr_result);
    int k = std::get<1>(rrqr_result);
    // Step 2: Truncate QR Result to rank k
    auto tqr = truncate_qr_result(A_qr, k);
    auto Q_trunc = tqr.first;
    auto R_trunc = tqr.second;
    // TODO

    // Step 3: Compute SVD of R_trunc
    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(R_trunc.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> U = Q_trunc * svd.matrixV();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V = svd.matrixU().transpose();
    Eigen::VectorXd s = svd.singularValues();

    return std::make_tuple(U, s, V.transpose());
}


template <typename T>
std::tuple<std::pair<T, T>, std::pair<T, T>, std::pair<T, T>> svd2x2(T f, T g, T h) {
    T fa = T(std::abs((double)f));
    T ga = T(std::abs((double)g));
    T ha = T(std::abs((double)h));

    T cu, su, smax, smin, cv, sv;

    if (fa < ha) {
        // switch h <-> f, cu <-> sv, cv <-> su
        //std::tie(std::tie(sv, cv), std::tie(smax, smin), std::tie(su, cu)) = svd2x2(h, g, f);
        auto svd_result = svd2x2(h, g, f);
        sv = std::get<0>(std::get<0>(svd_result));
        cv = std::get<1>(std::get<0>(svd_result));
        smax = std::get<0>(std::get<1>(svd_result));
        smin = std::get<1>(std::get<1>(svd_result));
        su = std::get<0>(std::get<2>(svd_result));
        cu = std::get<1>(std::get<2>(svd_result));

    } else if (ga == T(0)) {
        // already diagonal, fa > ha
        smax = fa;
        smin = ha;
        cv = cu = T(1);
        sv = su = T(0);
    } else if (fa < std::numeric_limits<T>::epsilon() * ga) {
        // ga is very large
        smax = ga;
        if (ha > T(1)) {
            smin = fa / (ga / ha);
        } else {
            smin = (fa / ga) * ha;
        }
        cv = f / g;
        sv = T(1);
        cu = T(1);
        su = h / g;
    } else {
        // normal case
        T fmh = fa - ha;
        T d = fmh / fa;
        T q = g / f;
        T s = T(2) - d;
        T spq = T(std::hypot((double)q, (double)s));
        T dpq = T(std::hypot((double)d, (double)q));
        T a = (spq + dpq) / T(2);
        smax = T(std::abs((double)(fa * a)));
        smin = T(std::abs((double)(ha / a)));

        T tmp = (q / (spq + s) + q / (dpq + d)) * (T(1) + a);
        T tt = T(std::hypot((double)tmp, (double)2));
        cv = T(2) / tt;
        sv = tmp / tt;
        cu = (cv + sv * q) / a;
        su = ((h / f) * sv) / a;
    }

    return std::make_tuple(std::make_pair(cu, su), std::make_pair(smax, smin), std::make_pair(cv, sv));
}

template <typename T>
std::tuple<std::tuple<T, T>, T> givens_params(T f, T g) {
    if (g == 0) {
        return {{T(1), T(0)}, f};
    } else if (f == 0) {
        return {{T(0), T(std::copysign(1.0, (double)g))}, T(std::abs((double)g))};
    } else {
        T r = T(std::copysign(std::hypot((double)f, (double)g), (double)f));
        T c = f / r;
        T s = g / r;
        return {{c, s}, r};
    }
}

template <typename T>
std::pair<T, T> givens_lmul(T c, T s, T x, T y) {
    T a = c * x + s * y;
    T b = c * y - s * x;
    return {a, b};
}

template <typename T>
std::tuple<std::tuple<T, T>, std::tuple<T, T>, std::tuple<T, T>> svd2x2(T a11, T a12, T a21, T a22) {
    T abs_a12 = std::abs((double)(a12));
    T abs_a21 = std::abs((double)(a21));

    if (a21 == 0) {
        return svd2x2<T>(a11, a12, a22);
    } else if (abs_a12 < abs_a21) {
        auto svd_result = svd2x2(a11, a21, a12, a22);
        auto cu = std::get<0>(std::get<0>(svd_result));
        auto su = std::get<1>(std::get<0>(svd_result));
        auto smax_smin = std::get<1>(svd_result);
        auto cv = std::get<0>(std::get<2>(svd_result));
        auto sv = std::get<1>(std::get<2>(svd_result));
        return std::make_tuple(std::make_pair(cv, sv), smax_smin, std::make_pair(cu, su));
    } else {
        auto rot = givens_params<T>(a11, a21);
        auto cx_sx = std::get<0>(rot);
        auto cx = std::get<0>(cx_sx);
        auto sx = std::get<1>(cx_sx);
        auto rx = std::get<1>(rot);
        a11 = rx;
        a21 = 0;
        std::tie(a12, a22) = givens_lmul(cx, sx, a12, a22);

        // auto [cu, su, smax, smin, cv, sv] = svd2x2(a11, a12, a22);
        auto svd_result = svd2x2(a11, a12, a22);
        auto cu = std::get<0>(std::get<0>(svd_result));
        auto su = std::get<1>(std::get<0>(svd_result));
        auto smax = std::get<0>(std::get<1>(svd_result));
        auto smin = std::get<1>(std::get<1>(svd_result));
        auto cv = std::get<0>(std::get<2>(svd_result));
        auto sv = std::get<1>(std::get<2>(svd_result));
        std::tie(cu, su) = givens_lmul(cx, -sx, cu, su);

        return std::make_tuple(std::make_tuple(cu, su), std::make_tuple(smax, smin), std::make_tuple(cu, sv));
    }
}


template <typename T>
T jacobi_sweep(Matrix<T, Dynamic, Dynamic>& U, Matrix<T, Dynamic, Dynamic>& VT) {
    int ii = U.rows();
    int jj = U.cols();
    if (ii < jj) {
        throw std::invalid_argument("matrix must be 'tall'");
    }
    if (VT.rows() != jj) {
        throw std::invalid_argument("U and VT must be compatible");
    }

    T offd = T(0);
    for (int i = 0; i < ii; ++i) {
        for (int j = i + 1; j < jj; ++j) {
            T Hii = U.col(i).squaredNorm();
            T Hij = U.col(i).dot(U.col(j));
            T Hjj = U.col(j).squaredNorm();
            offd += Hij * Hij;

            auto svd_result = svd2x2(Hii, Hij, Hij, Hjj);
            T cv = std::get<0>(std::get<2>(svd_result));
            T sv = std::get<1>(std::get<2>(svd_result));
            Eigen::JacobiRotation<T> rot(cv, sv);
            VT.applyOnTheLeft(j, i, rot);
            U.applyOnTheRight(i, j, rot.transpose());
        }
    }
    // TODO: Fix me:
    return T(std::sqrt<double>((double)offd));
}

template <typename T>
SVDResult<T> svd_jacobi(Matrix<T, Dynamic, Dynamic>& U, T rtol = std::numeric_limits<T>::epsilon(), int maxiter = 20) {
    int m = U.rows();
    int n = U.cols();
    if (m < n) {
        throw std::invalid_argument("matrix must be 'tall'");
    }

    Matrix<T, Dynamic, Dynamic> VT = Matrix<T, Dynamic, Dynamic>::Identity(n, n);
    T Unorm = U.topLeftCorner(n, n).norm();

    for (int iter = 0; iter < maxiter; ++iter) {
        T offd = jacobi_sweep(U, VT);
        if (offd < rtol * Unorm) {
            break;
        }
    }

    Vector<T, Dynamic> s = U.colwise().norm();
    U.array().rowwise() /= s.transpose().array();

    return SVDResult<T> {U, s, VT};
}