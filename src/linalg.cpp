#include <Eigen/Dense>

namespace sparseir {

Eigen::MatrixXd pinv(const Eigen::MatrixXd &A, double tolerance = 1e-6)
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