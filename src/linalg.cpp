#include <Eigen/Dense>

#include "sparseir/_linalg.hpp"
#include "sparseir/impl/linalg_impl.hpp"

namespace sparseir {

Eigen::MatrixXd pinv(const Eigen::MatrixXd &A, double tolerance)
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

// Explicit template instantiations
template void lmul<double>(const QRPackedQ<double>, Eigen::MatrixX<double>&);
template void lmul<xprec::DDouble>(const QRPackedQ<xprec::DDouble>, Eigen::MatrixX<xprec::DDouble>&);

template void mul<double>(Eigen::MatrixX<double>&, const QRPackedQ<double>&, const Eigen::MatrixX<double>&);
template void mul<xprec::DDouble>(Eigen::MatrixX<xprec::DDouble>&, const QRPackedQ<xprec::DDouble>&, const Eigen::MatrixX<xprec::DDouble>&);

} // namespace sparseir