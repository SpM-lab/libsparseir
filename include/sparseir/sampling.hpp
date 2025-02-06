#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sparseir {

// Define WorkSize struct
// struct WorkSize
//{
// Eigen::Index rows;
// Eigen::Index cols;
//
// WorkSize(Eigen::Index r, Eigen::Index c) : rows(r), cols(c) { }
//
// Eigen::Index prod() const { return rows * cols; }
// Eigen::Index size() const { return rows; }
// Eigen::Index dimensions() const { return cols; }
//};

template <int N>
Eigen::array<int, N> getperm(int src, int dst)
{
    Eigen::array<int, N> perm;
    if (src == dst) {
        for (int i = 0; i < N; ++i) {
            perm[i] = i;
        }
        return perm;
    }

    int pos = 0;
    for (int i = 0; i < N; ++i) {
        if (i == dst) {
            perm[i] = src;
        } else {
            // src の位置をスキップ
            if (pos == src)
                ++pos;
            perm[i] = pos;
            ++pos;
        }
    }
    return perm;
}

// movedim: テンソル arr の次元 src を次元 dst
// に移動する（他の次元の順序はそのまま）
template <typename T, int N>
Eigen::Tensor<T, N> movedim(const Eigen::Tensor<T, N> &arr, int src, int dst)
{
    if (src == dst) {
        return arr;
    }
    auto perm = getperm<N>(src, dst);
    return arr.shuffle(perm);
}

template <typename T, typename S, int N>
Eigen::Matrix<decltype(T() * S()), Eigen::Dynamic, Eigen::Dynamic>
_fit_impl_first_dim(const Eigen::JacobiSVD<Eigen::MatrixX<S>> &svd,
                    const Eigen::MatrixX<T> &B)
{
    using ResultType = decltype(T() * S());
    Eigen::Matrix<ResultType, Eigen::Dynamic, Eigen::Dynamic> UHB =
        svd.matrixU().adjoint() * B;

    // Apply inverse singular values to the rows of UHB
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        UHB.row(i) /= ResultType(svd.singularValues()(i));
    }

    return svd.matrixV() * UHB;
}

template <typename T, typename S, int N>
Eigen::Tensor<decltype(T() * S()), N>
fit_impl(const Eigen::JacobiSVD<Eigen::MatrixX<S>> &svd,
         const Eigen::Tensor<T, N> &arr, int dim)
{
    if (dim < 0 || dim >= N) {
        throw std::domain_error("Dimension must be in [0, N).");
    }

    // First move the dimension to the first
    auto arr_ = movedim(arr, dim, 0);

    // Create a view of the tensor as a matrix
    Eigen::MatrixX<T> arr_view = Eigen::Map<Eigen::MatrixX<T>>(
        arr_.data(), arr_.dimension(0), arr_.size() / arr_.dimension(0));

    Eigen::MatrixX<T> result = _fit_impl_first_dim<T, S, N>(svd, arr_view);

    // Copy the result to a tensor
    Eigen::array<Eigen::Index, N> dims;
    dims[0] = result.rows();
    for (int i = 1; i < N; ++i) {
        dims[i] = arr_.dimension(i);
    }
    Eigen::Tensor<T, N> result_tensor(dims);
    std::copy(result.data(), result.data() + result.size(),
              result_tensor.data());

    return movedim(result_tensor, 0, dim);
}

template <typename S>
class AbstractSampling {
public:
    virtual ~AbstractSampling() = default;

    // Evaluate the basis coefficients at sampling points
    template <typename T, int N>
    Eigen::Tensor<T, N> evaluate(const Eigen::Tensor<T, N> &al,
                                 int dim = 0) const
    {
        if (dim < 0 || dim >= N) {
            throw std::runtime_error(
                "evaluate: dimension must be in [0..N). Got dim=" +
                std::to_string(dim));
        }

        if (get_matrix().cols() != al.dimension(dim)) {
            throw std::runtime_error(
                "Mismatch: matrix.cols()=" +
                std::to_string(get_matrix().cols()) + ", but al.dimension(" +
                std::to_string(dim) + ")=" + std::to_string(al.dimension(dim)));
        }

        // Convert matrix to tensor
        Eigen::Tensor<double, 2> matrix_tensor =
            Eigen::TensorMap<Eigen::Tensor<const double, 2>>(
                get_matrix().data(), get_matrix().rows(), get_matrix().cols());

        // Specify contraction dimensions
        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
            Eigen::IndexPair<int>(1, dim)};

        // Perform contraction
        Eigen::Tensor<T, N> temp = matrix_tensor.contract(al, contract_dims);

        return movedim(temp, 0, dim);
    }

    // template <typename T, int N>
    // size_t workarrlength(const Eigen::Tensor<T, N> &ax, int dim) const
    //{
    // auto svd = get_matrix_svd();
    // return svd.singularValues().size() * (ax.size() / ax.dimension(dim));
    //}

    // Fit values at sampling points to basis coefficients
    template <typename T, int N>
    Eigen::Tensor<T, N> fit(const Eigen::Tensor<T, N> &ax, int dim = 0) const
    {
        if (dim < 0 || dim >= N) {
            throw std::runtime_error(
                "fit: dimension must be in [0..N). Got dim=" +
                std::to_string(dim));
        }
        auto svd = get_matrix_svd();
        return fit_impl<T, double, N>(svd, ax, dim);
    }

    // Get the sampling points
    virtual const Eigen::VectorXd &sampling_points() const = 0;
    virtual Eigen::MatrixXd get_matrix() const = 0;
    virtual Eigen::JacobiSVD<Eigen::MatrixXd> get_matrix_svd() const = 0;
};
// Helper function declarations
// Forward declarations
template <typename S>
class TauSampling;

template <typename S>
inline Eigen::MatrixXd
eval_matrix(const TauSampling<S> *tau_sampling,
            const std::shared_ptr<FiniteTempBasis<S>> &basis,
            const Eigen::VectorXd &x)
{
    // Initialize matrix with correct dimensions
    Eigen::MatrixXd matrix(x.size(), basis->size());

    // Evaluate basis functions at sampling points
    auto u_eval = basis->u(x);
    // Transpose and scale by singular values
    matrix = u_eval.transpose();

    return matrix;
}

template <typename S>
class TauSampling : public AbstractSampling<S> {
public:
    TauSampling(const std::shared_ptr<FiniteTempBasis<S>> &basis,
                bool factorize = true)
        : basis_(basis)
    {
        // Get default sampling points from basis
        sampling_points_ = basis_->default_tau_sampling_points();

        // Ensure matrix dimensions are correct
        if (sampling_points_.size() == 0) {
            throw std::runtime_error("No sampling points generated");
        }

        // Initialize evaluation matrix with correct dimensions
        matrix_ = eval_matrix(this, basis_, sampling_points_);
        // Check matrix dimensions
        if (matrix_.rows() != sampling_points_.size() ||
            matrix_.cols() != basis_->size()) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points_.size()) +
                                     "x" + std::to_string(basis_->size()));
        }

        // Initialize SVD
        if (factorize) {
            matrix_svd_ = Eigen::JacobiSVD<Eigen::MatrixXd>(
                matrix_, Eigen::ComputeFullU | Eigen::ComputeFullV);
        }
    }

    Eigen::MatrixXd get_matrix() const override { return matrix_; }

    const Eigen::VectorXd &sampling_points() const override
    {
        return sampling_points_;
    }

    const Eigen::VectorXd &tau() const { return sampling_points_; }
    Eigen::JacobiSVD<Eigen::MatrixXd> get_matrix_svd() const override
    {
        return matrix_svd_;
    }

private:
    std::shared_ptr<FiniteTempBasis<S>> basis_;
    Eigen::VectorXd sampling_points_;
    Eigen::MatrixXd matrix_;
    Eigen::JacobiSVD<Eigen::MatrixXd> matrix_svd_;
};

} // namespace sparseir
