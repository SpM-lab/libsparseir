#include "sparseir/kernel.hpp"
#include "sparseir/impl/kernel_impl.hpp"

namespace sparseir {

using xprec::DDouble;

// Explicit instantiation for T = double
template Eigen::MatrixX<double> matrix_from_gauss(const std::shared_ptr<AbstractKernel> &kernel,
                                                 const Rule<double> &gauss_x,
                                                 const Rule<double> &gauss_y);

// Explicit instantiation for T = DDouble
template Eigen::MatrixX<DDouble> matrix_from_gauss(const std::shared_ptr<AbstractKernel> &kernel,
                                                  const Rule<DDouble> &gauss_x,
                                                  const Rule<DDouble> &gauss_y);

// Explicit instantiation for T = double
template std::vector<double> symm_segments(const std::vector<double> &x);

// Explicit instantiation for T = DDouble  
template std::vector<DDouble> symm_segments(const std::vector<DDouble> &x);

}