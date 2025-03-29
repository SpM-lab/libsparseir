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
void lmul(const QRPackedQ<T> Q, Eigen::MatrixX<T> &B);

template <typename T>
void mul(Eigen::MatrixX<T> &C, const QRPackedQ<T> &Q,
         const Eigen::MatrixX<T> &B);


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

Eigen::MatrixXd pinv(const Eigen::MatrixXd &A, double tolerance = 1e-6);


} // namespace sparseir


//#include "impl/linalg_impl.hpp"