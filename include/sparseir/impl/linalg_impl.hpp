#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "xprec/ddouble.hpp"

#include "sparseir/_linalg.hpp"

namespace sparseir {

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

} // namespace sparseir