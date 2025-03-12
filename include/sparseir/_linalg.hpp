#pragma once

#include <iostream>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "xprec/ddouble.hpp"

namespace sparseir {

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixX;
using Eigen::Vector;
using Eigen::VectorX;

template <typename Derived>
int argmax(const Eigen::MatrixBase<Derived> &vec)
{
    // Ensure this function is only used for 1D Eigen column vectors
    static_assert(Derived::ColsAtCompileTime == 1,
                  "argmax function is only for Eigen column vectors.");

    int maxIndex = 0;
    auto maxValue = vec(0);

    for (int i = 1; i < vec.size(); ++i) {
        if (vec(i) > maxValue) {
            maxValue = vec(i);
            maxIndex = i;
        }
    }

    return maxIndex;
}

/*
This function ports Julia's implementation of the `invperm` function to C++.
*/
inline Eigen::VectorXi invperm(const Eigen::VectorXi &a)
{
    int n = a.size();
    Eigen::VectorXi b(n);
    b.setConstant(-1);

    for (int i = 0; i < n; i++) {
        int j = a(i);
        if ((0 <= j < n) && b(j) == -1) {
            std::invalid_argument("invalid permutation");
        }
        b(j) = i;
    }
    return b;
}

template <typename T>
struct SVDResult
{
    Matrix<T, Dynamic, Dynamic> U;
    Vector<T, Dynamic> s;
    Matrix<T, Dynamic, Dynamic> V;
};

template <typename T>
struct QRPivoted
{
    Matrix<T, Dynamic, Dynamic> factors;
    Vector<T, Dynamic> taus;
    Vector<int, Dynamic> jpvt;
};

template <typename T>
struct QRPackedQ
{
    Eigen::MatrixX<T> factors;
    Eigen::Matrix<T, Eigen::Dynamic, 1> taus;
};

template <typename T>
void lmul(const QRPackedQ<T> Q, Eigen::MatrixX<T> &B)
{
    Eigen::MatrixX<T> A_factors = Q.factors;
    Eigen::VectorX<T> A_tau = Q.taus;
    int mA = A_factors.rows();
    int nA = A_factors.cols();
    int mB = B.rows();
    int nB = B.cols();

    if (mA != mB) {
        throw std::invalid_argument("DimensionMismatch: matrix A has different "
                                    "dimensions than matrix B");
    }

    for (int k = std::min(mA, nA) - 1; k >= 0; --k) {
        for (int j = 0; j < nB; ++j) {
            T vBj = B(k, j);
            for (int i = k + 1; i < mB; ++i) {
                vBj += A_factors(i, k) * B(i, j);
            }
            vBj = A_tau(k) * vBj;
            B(k, j) -= vBj;
            for (int i = k + 1; i < mB; ++i) {
                B(i, j) -= A_factors(i, k) * vBj;
            }
        }
    }
}

template <typename T>
void mul(Eigen::MatrixX<T> &C, const QRPackedQ<T> &Q,
         const Eigen::MatrixX<T> &B)
{

    int mB = B.rows();
    int nB = B.cols();
    int mC = C.rows();
    int nC = C.cols();

    if (nB != nC) {
        throw std::invalid_argument(
            "DimensionMismatch: number of columns in B and C do not match");
    }

    if (mB < mC) {
        C.topRows(mB) = B;
        C.bottomRows(mC - mB).setZero();
        lmul<T>(Q, C);
    } else {
        C = B;
        lmul<T>(Q, C);
    }
}

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

    QRPackedQ(const Matrix<T, Dynamic, Dynamic>& factors, const Matrix<T,
Dynamic, 1>& taus) : factors(factors), taus(taus) {}
};
*/

// TODO: FIX THIS
template <typename T>
MatrixX<T> getPropertyP(const QRPivoted<T> &F)
{
    int n = F.factors.cols();

    MatrixX<T> P = MatrixX<T>::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        P(F.jpvt[i], i) = 1;
    }
    return P;
}

template <typename T>
QRPackedQ<T> getPropertyQ(const QRPivoted<T> &F)
{
    return QRPackedQ<T>{F.factors, F.taus};
}

template <typename T>
MatrixX<T> getPropertyR(const QRPivoted<T> &F)
{
    int m = F.factors.rows();
    int n = F.factors.cols();

    MatrixX<T> upper = MatrixX<T>::Zero(std::min(m, n), n);

    for (int i = 0; i < upper.rows(); ++i) {
        for (int j = i; j < upper.cols(); ++j) {
            upper(i, j) = F.factors(i, j);
        }
    }
    return upper;
}

// General template for _copysign, handles standard floating-point types like
// double and float
inline double _copysign(double x, double y) { return std::copysign(x, y); }

// Specialization for xprec::DDouble type, assuming xprec::copysign is defined
inline xprec::DDouble _copysign(xprec::DDouble x, xprec::DDouble y)
{
    return xprec::copysign(x, y);
}

/*
This implementation is based on Julia's LinearAlgebra.refrector! function.

Elementary reflection similar to LAPACK. The reflector is not Hermitian but
ensures that tridiagonalization of Hermitian matrices become real. See lawn72
*/
template <typename Derived>
typename Derived::Scalar reflector(Eigen::MatrixBase<Derived> &x)
{
    using T = typename Derived::Scalar;

    int n = x.size();
    if (n == 0)
        return T(0);

    T xi1 = x(0);       // First element of x
    T normu = x.norm(); // Norm of vector x
    if (normu == T(0))
        return T(0);

    // Calculate ν using copysign, which gives normu with the sign of xi1
    T nu = _copysign(normu, xi1);
    xi1 += nu;  // Update xi1
    x(0) = -nu; // Set first element to -ν

    // Divide remaining elements by the new xi1 value
    for (int i = 1; i < n; ++i) {
        x(i) /= xi1;
    }

    return xi1 / nu;
}

/*
This implementation is based on Julia's LinearAlgebra.reflectorApply! function.

    reflectorApply!(x, τ, A)

Multiplies `A` in-place by a Householder reflection on the left. It is
equivalent to `A .= (I - conj(τ)*[1; x[2:end]]*[1; x[2:end]]')*A`.
*/
template <typename T>
void reflectorApply(
    Eigen::VectorBlock<Eigen::Block<Eigen::MatrixX<T>, -1, 1, true>, -1> &x,
    T tau, Eigen::Block<Eigen::MatrixX<T>> &A)
{
    int m = A.rows();
    int n = A.cols();

    if (x.size() != m) {
        throw std::invalid_argument(
            "Reflector length must match the first dimension of matrix A");
    }

    if (m == 0)
        return;

    // Loop over each column of A
    for (int j = 0; j < n; ++j) {
        // Equivalent to `Aj = view(A, 2:m, j)` and `xj = view(x, 2:m)`
        auto Aj = A.block(1, j, m - 1, 1).reshaped();
        auto xj = x.tail(m - 1);

        // We expect tau to be real, so we use conj(tau) = tau
        T conj_tau = tau;
        // Compute vAj = conj(tau) * (A(0, j) + xj.dot(Aj));
        T vAj = conj_tau * (A(0, j) + xj.dot(Aj));

        // Update A(0, j)
        A(0, j) -= vAj;

        // Apply axpy operation: Aj -= vAj * xj
        Aj.noalias() -= vAj * xj;
    }
}

/*
A will be modified in-place.
*/
template <typename T>
std::pair<QRPivoted<T>, int> rrqr(MatrixX<T> &A,
                                  T rtol = std::numeric_limits<T>::epsilon())
{
    using std::abs;
    using std::sqrt;

    int m = A.rows();
    int n = A.cols();
    int k = std::min(m, n);
    int rk = k;
    Vector<int, Dynamic> jpvt = Vector<int, Dynamic>::LinSpaced(n, 0, n - 1);
    Vector<T, Dynamic> taus(k);

    Vector<T, Dynamic> xnorms = A.colwise().norm();
    Vector<T, Dynamic> pnorms = xnorms;
    T sqrteps = sqrt(std::numeric_limits<T>::epsilon());

    for (int i = 0; i < k; ++i) {

        int pvt = argmax(pnorms.tail(n - i)) + i;
        if (i != pvt) {
            std::swap(jpvt[i], jpvt[pvt]);
            std::swap(xnorms[pvt], xnorms[i]);
            std::swap(pnorms[pvt], pnorms[i]);
            A.col(i).swap(A.col(pvt)); // swapcols!
        }

        auto Ainp = A.col(i).tail(m - i);
        T tau_i = reflector(Ainp);

        taus[i] = tau_i;

        auto block = A.bottomRightCorner(m - i, n - (i + 1));
        reflectorApply<T>(Ainp, tau_i, block);

        for (int j = i + 1; j < n; ++j) {
            T temp = abs((A(i, j))) / pnorms[j];
            temp = std::max<T>(T(0), (T(1) + temp) * (T(1) - temp));
            // abs2
            T temp2 = temp * (pnorms(j) / xnorms(j)) * (pnorms(j) / xnorms(j));
            if (temp2 < sqrteps) {
                auto recomputed = A.col(j).tail(m - i - 1).norm();
                pnorms(j) = recomputed;
                xnorms(j) = recomputed;
            } else {
                pnorms(j) = pnorms(j) * sqrt(temp);
            }
        }

        if (abs(A(i, i)) < rtol * abs((A(0, 0)))) {
            A.bottomRightCorner(m - i, n - i).setZero();
            taus.tail(k - i).setZero();
            rk = i;
            break;
        }
    }

    return {QRPivoted<T>{A, taus, jpvt}, rk};
}

/*
template <typename T>
std::tuple<Eigen::MatrixX<T>, Eigen::MatrixX<T>, std::vector<int>>
rrqr(Eigen::MatrixX<T>& A, T rtol = std::numeric_limits<T>::epsilon()) {
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

template <typename Derived>
Eigen::MatrixBase<Derived> triu(Eigen::MatrixBase<Derived> &M)
{
    using T = typename Derived::Scalar;
    int m = M.rows();
    int n = M.cols();
    for (int j = 0; j < std::min(n, m); ++j) {
        for (int i = std::max(0, j + 1); i < m; ++i) {
            M(i, j) = T(0);
        }
    }
    return M;
}

template <typename T>
std::pair<MatrixX<T>, MatrixX<T>> truncate_qr_result(QRPivoted<T> &qr, int k)
{
    int m = qr.factors.rows();
    int n = qr.factors.cols();
    if (k < 0 || k > std::min(m, n)) {
        throw std::domain_error("Invalid rank, must be in [0, " +
                                std::to_string(std::min(m, n)) + "]");
    }

    // Extract Q matrix

    MatrixX<T> k_factors = qr.factors.topLeftCorner(qr.factors.rows(), k);
    MatrixX<T> k_taus = qr.taus.head(k);
    auto Qfull = QRPackedQ<T>{k_factors, k_taus};

    MatrixX<T> Q = Eigen::MatrixX<T>::Identity(m, k);
    lmul<T>(Qfull, Q);
    // Extract R matrix
    auto R = qr.factors.topRows(k);
    // inline implementation of triu(qr.factors.topRows(k))
    for (int j = 0; j < R.rows(); ++j) {
        for (int i = j + 1; i < R.rows(); ++i) {
            R(i, j) = T(0);
        }
    }
    return std::make_pair(Q, R);
}

/*
template <typename T>
std::pair<Eigen::MatrixX<T>, Eigen::MatrixX<T>>
truncate_qr_result(const Eigen::HouseholderQR<Eigen::MatrixX<T>>& qr, int k) {
    int m = qr.matrixQR().rows();
    int n = qr.matrixQR().cols();

    // Ensure k is within the valid range [0, min(m, n)]
    if (!(0 <= k && k <= std::min(m, n))) {
        throw std::domain_error("Invalid rank, must be in [0, min(m, n)]");
    }

    // Extract the first k columns of Q
    Eigen::MatrixX<T> Qfull = qr.householderQ() * Eigen::MatrixX<T>::Identity(m,
k);

    // Extract the upper triangular part of the first k rows of R
    Eigen::MatrixX<T> R = qr.matrixQR().topLeftCorner(k, n).template
triangularView<Eigen::Upper>();

    return std::make_pair(Qfull, R);
}
*/

/*
template <typename T>
SVDResult<T> tsvd(Matrix<T, Dynamic, Dynamic>& A, double rtol =
std::numeric_limits<double>::epsilon()) { auto A_qr_k = rrqr<T>(A, rtol);
    // FIX ME
    auto QR_result = truncate_qr_result<T>(A_qr_k.first, A_qr_k.second);
    auto Q = QR_result.first;
    auto R = QR_result.second;
    Eigen::JacobiSVD<Matrix<T, Dynamic, Dynamic>> svd(R.transpose(),
Eigen::ComputeThinU | Eigen::ComputeThinV); Matrix<T, Dynamic, Dynamic> U = Q *
svd.matrixV(); Matrix<T, Dynamic, Dynamic> V = svd.matrixU().transpose();
    Vector<T, Dynamic> s = svd.singularValues();

    return SVDResult<T> {U, s, V};
}
*/

// Swap columns of a matrix A
template <typename T>
void swapCols(Eigen::MatrixX<T> &A, int i, int j)
{
    if (i != j) {
        A.col(i).swap(A.col(j));
    }
}

// Truncate RRQR result to low rank
template <typename T>
std::pair<Eigen::MatrixX<T>, Eigen::MatrixX<T>>
truncateQRResult(const Eigen::MatrixX<T> &Q, const Eigen::MatrixX<T> &R, int k)
{
    int m = Q.rows();
    int n = R.cols();

    if (k < 0 || k > std::min(m, n)) {
        throw std::domain_error("Invalid rank, must be in [0, min(m, n)]");
    }

    Eigen::MatrixX<T> Q_trunc = Q.leftCols(k);
    Eigen::MatrixX<T> R_trunc = R.topLeftCorner(k, n);
    return std::make_pair(Q_trunc, R_trunc);
}

// Truncated SVD (TSVD)
template <typename T>
std::tuple<Eigen::MatrixX<T>, Eigen::VectorX<T>, Eigen::MatrixX<T>>
tsvd(const Eigen::MatrixX<T> &A, T rtol = std::numeric_limits<T>::epsilon())
{
    // Step 1: Apply RRQR to A
    QRPivoted<T> A_qr;
    int k;
    Eigen::MatrixX<T> A_ = A; // create a copy of A
    std::tie(A_qr, k) = rrqr<T>(A_, rtol);
    // Step 2: Truncate QR Result to rank k
    auto tqr = truncate_qr_result<T>(A_qr, k);
    auto p = A_qr.jpvt;
    auto Q_trunc = tqr.first;
    auto R_trunc = tqr.second;
    // TODO

    // Step 3: Compute SVD of R_trunc

    // Eigen::JacobiSVD<Eigen::MatrixX<T>> svd(R_trunc.transpose(),
    // Eigen::ComputeThinU | Eigen::ComputeThinV);
    //  There seems to be a bug in the latest version of Eigen3.
    //  Please first construct a Jacobi SVD and then compare the results.
    //  Do not use the svd_jacobi function directly.
    //  Better to write a wrrapper function for the SVD.
    Eigen::JacobiSVD<decltype(R_trunc)> svd;

    // The following comment is taken from Julia's implementation
    // # RRQR is an excellent preconditioner for Jacobi. One should then perform
    // # Jacobi on RT
    svd.compute(R_trunc.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Reconstruct A from QR factorization
    Eigen::PermutationMatrix<Dynamic, Dynamic> perm(p.size());
    perm.indices() = p;

    Eigen::MatrixX<T> U = Q_trunc * svd.matrixV();
    // implement invperm
    Eigen::MatrixX<T> V = (perm * svd.matrixU());

    Eigen::VectorX<T> s = svd.singularValues();
    // TODO: Create a return type for truncated SVD
    return std::make_tuple(U, s, V);
}

template <typename T>
std::tuple<std::pair<T, T>, std::pair<T, T>, std::pair<T, T>> svd2x2(T f, T g,
                                                                     T h)
{
    T fa = T(std::abs((double)f));
    T ga = T(std::abs((double)g));
    T ha = T(std::abs((double)h));

    T cu, su, smax, smin, cv, sv;

    if (fa < ha) {
        // switch h <-> f, cu <-> sv, cv <-> su
        // std::tie(std::tie(sv, cv), std::tie(smax, smin), std::tie(su, cu)) =
        // svd2x2(h, g, f);
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

    return std::make_tuple(std::make_pair(cu, su), std::make_pair(smax, smin),
                           std::make_pair(cv, sv));
}

template <typename T>
std::tuple<std::tuple<T, T>, T> givens_params(T f, T g)
{
    if (g == 0) {
        return {{T(1), T(0)}, f};
    } else if (f == 0) {
        return {{T(0), T(std::copysign(1.0, (double)g))},
                T(std::abs((double)g))};
    } else {
        T r = T(std::copysign(std::hypot((double)f, (double)g), (double)f));
        T c = f / r;
        T s = g / r;
        return {{c, s}, r};
    }
}

template <typename T>
std::pair<T, T> givens_lmul(T c, T s, T x, T y)
{
    T a = c * x + s * y;
    T b = c * y - s * x;
    return {a, b};
}

template <typename T>
std::tuple<std::tuple<T, T>, std::tuple<T, T>, std::tuple<T, T>>
svd2x2(T a11, T a12, T a21, T a22)
{
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
        return std::make_tuple(std::make_pair(cv, sv), smax_smin,
                               std::make_pair(cu, su));
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

        return std::make_tuple(std::make_tuple(cu, su),
                               std::make_tuple(smax, smin),
                               std::make_tuple(cu, sv));
    }
}

template <typename T>
T jacobi_sweep(Matrix<T, Dynamic, Dynamic> &U, Matrix<T, Dynamic, Dynamic> &VT)
{
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
SVDResult<T> svd_jacobi(Matrix<T, Dynamic, Dynamic> &U,
                        T rtol = std::numeric_limits<T>::epsilon(),
                        int maxiter = 20)
{
    int m = U.rows();
    int n = U.cols();
    if (m < n) {
        throw std::invalid_argument("matrix must be 'tall'");
    }

    Matrix<T, Dynamic, Dynamic> VT =
        Matrix<T, Dynamic, Dynamic>::Identity(n, n);
    T Unorm = U.topLeftCorner(n, n).norm();

    for (int iter = 0; iter < maxiter; ++iter) {
        T offd = jacobi_sweep(U, VT);
        if (offd < rtol * Unorm) {
            break;
        }
    }

    Vector<T, Dynamic> s = U.colwise().norm();
    U.array().rowwise() /= s.transpose().array();

    return SVDResult<T>{U, s, VT};
}

inline Eigen::MatrixXd pinv(const Eigen::MatrixXd &A, double tolerance = 1e-6)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU |
                                                 Eigen::ComputeThinV);
    const auto &singularValues = svd.singularValues();
    Eigen::MatrixXd singularValuesInv =
        Eigen::MatrixXd::Zero(A.cols(), A.rows());
    for (int i = 0; i < singularValues.size(); ++i) {
        if (singularValues(i) > tolerance)
            singularValuesInv(i, i) = 1.0 / singularValues(i);
    }
    return svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
}

} // namespace sparseir