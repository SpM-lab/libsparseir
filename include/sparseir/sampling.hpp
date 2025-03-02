#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>
#include <complex>
#include <tuple>

namespace sparseir {

// Forward declarations
template <typename S> class FiniteTempBasis;
template <typename S> class DiscreteLehmannRepresentation;

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
    for (int i = 0; i < svd.singularValues().size(); ++i)
    {
        UHB.row(i) /= ResultType(svd.singularValues()(i));
    }
    return svd.matrixV() * UHB;
}

inline Eigen::MatrixXcd _fit_impl_first_dim_split_svd(const Eigen::JacobiSVD<Eigen::MatrixXcd> &svd,
                    const Eigen::MatrixXcd &B, bool has_zero)
{
    auto U = svd.matrixU();

    Eigen::Index U_halfsize = U.rows() % 2 == 0 ? U.rows() / 2 : U.rows() / 2 + 1;

    Eigen::MatrixXcd U_realT;
    U_realT = U.block(0, 0, U_halfsize, U.cols()).transpose();

    // Create a properly sized matrix first
    Eigen::MatrixXcd U_imag = Eigen::MatrixXcd::Zero(U_halfsize, U.cols());

    // Get the blocks we need
    if (has_zero) {
        U_imag = Eigen::MatrixXcd::Zero(U_halfsize, U.cols());
        auto U_imag_ = U.block(U_halfsize, 0, U_halfsize - 1, U.cols());
        auto U_imag_1 = U.block(0, 0, 1, U.cols());

        // Now do the assignments
        U_imag.topRows(1) = U_imag_1;
        U_imag.bottomRows(U_imag_.rows()) = U_imag_;
    } else {
        U_imag = U.block(U_halfsize, 0, U_halfsize, U.cols());
    }

    auto U_imagT = U_imag.transpose();

    Eigen::MatrixXcd UHB = U_realT * B.real();
    UHB += U_imagT * B.imag();

    // Apply inverse singular values to the rows of UHB
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        UHB.row(i) /= std::complex<double>(svd.singularValues()(i));
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

template <int N>
Eigen::Tensor<std::complex<double>, N>
fit_impl_split_svd(const Eigen::JacobiSVD<Eigen::MatrixXcd> &svd,
         const Eigen::Tensor<std::complex<double>, N> &arr, int dim, bool has_zero)
{
    if (dim < 0 || dim >= N) {
        throw std::domain_error("Dimension must be in [0, N).");
    }

    // First move the dimension to the first
    auto arr_ = movedim(arr, dim, 0);
    // Create a view of the tensor as a matrix
    Eigen::MatrixXcd arr_view = Eigen::Map<Eigen::MatrixXcd>(
        arr_.data(), arr_.dimension(0), arr_.size() / arr_.dimension(0));
    Eigen::MatrixXcd result = _fit_impl_first_dim_split_svd(svd, arr_view, has_zero);
    // Copy the result to a tensor
    Eigen::array<Eigen::Index, N> dims;
    dims[0] = result.rows();
    for (int i = 1; i < N; ++i) {
        dims[i] = arr_.dimension(i);
    }
    Eigen::Tensor<std::complex<double>, N> result_tensor(dims);
    std::copy(result.data(), result.data() + result.size(),
              result_tensor.data());

    return movedim(result_tensor, 0, dim);
}

template <typename S>
class AbstractSampling {
public:
    virtual ~AbstractSampling() = default;
};

// Helper function declarations
// Forward declarations
template <typename S>
class TauSampling;

template <typename S>
class MatsubaraSampling;

template <typename S>
class AugmentedBasis;

template <typename Basis>
inline Eigen::MatrixXd evaluate_u_at_x(const std::shared_ptr<Basis> &basis,
            const Eigen::VectorXd &x)
{
    return basis->u(x);
}

template <typename S>
inline Eigen::MatrixXd evaluate_u_at_x(const std::shared_ptr<AugmentedBasis<S>> &basis,
                                const Eigen::VectorXd &x)
{
    // Dereference the unique_ptr to access the AugmentedTauFunction
    return basis->u(x);
}

template <typename S, typename Basis>
inline Eigen::MatrixXd eval_matrix(const TauSampling<S> *tau_sampling,
            const std::shared_ptr<Basis> &basis,
            const Eigen::VectorXd &x)
{
    // Initialize matrix with correct dimensions
    Eigen::MatrixXd matrix(x.size(), basis->size());

    // Evaluate basis functions at sampling points
    auto u_eval = evaluate_u_at_x(basis, x);
    // Transpose and scale by singular values
    matrix = u_eval.transpose();

    return matrix;
}

template <typename S, typename T>
inline Eigen::VectorXcd
evaluate_uhat_at_x(const std::shared_ptr<FiniteTempBasis<S>> &basis,
                   const T &x)
{
    return basis->uhat(x);
}

template <typename S, typename T>
inline Eigen::VectorXcd
evaluate_uhat_at_x(const std::shared_ptr<AugmentedBasis<S>> &basis,
                   const T &x)
{
    return basis->uhat(x);
}

template <typename S, typename Base>
inline Eigen::MatrixXcd eval_matrix(const MatsubaraSampling<S> *matsubara_sampling,
                                   const std::shared_ptr<Base> &basis,
                                   const std::vector<MatsubaraFreq<S>> &sampling_points)
{
    Eigen::MatrixXcd m(basis->uhat.size(), sampling_points.size());
    for (int i = 0; i < sampling_points.size(); ++i) {
        m.col(i) = evaluate_uhat_at_x(basis, sampling_points[i]);
    }
    Eigen::MatrixXcd matrix = m.transpose();
    return matrix;
}

template <typename S>
class TauSampling : public AbstractSampling<S> {
private:
    std::shared_ptr<AbstractBasis<S>> basis_;
    Eigen::VectorXd sampling_points_;
    Eigen::MatrixXd matrix_;
    Eigen::JacobiSVD<Eigen::MatrixXd> matrix_svd_;

public:
    TauSampling(const std::shared_ptr<FiniteTempBasis<S>> &basis,
                bool factorize = true)
    {
        // Get default sampling points from basis
        sampling_points_ = basis->default_tau_sampling_points();

        // Ensure matrix dimensions are correct
        if (sampling_points_.size() == 0) {
            throw std::runtime_error("No sampling points generated");
        }

        // Initialize evaluation matrix with correct dimensions
        matrix_ = eval_matrix(this, basis, sampling_points_);
        // Check matrix dimensions
        if (matrix_.rows() != sampling_points_.size() ||
            matrix_.cols() != basis->size()) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points_.size()) +
                                     "x" + std::to_string(basis->size()));
        }

        // Initialize SVD
        if (factorize) {
            matrix_svd_ = Eigen::JacobiSVD<Eigen::MatrixXd>(
                matrix_, Eigen::ComputeThinU | Eigen::ComputeThinV);
        }

        basis_ = basis;
    }

    TauSampling(const std::shared_ptr<AugmentedBasis<S>> &basis,
                bool factorize = true)
    {
        // Get default sampling points from basis
        sampling_points_ = basis->default_tau_sampling_points();

        // Ensure matrix dimensions are correct
        if (sampling_points_.size() == 0) {
            throw std::runtime_error("No sampling points generated");
        }

        // Initialize evaluation matrix with correct dimensions
        matrix_ = eval_matrix(this, basis, sampling_points_);
        // Check matrix dimensions
        if (matrix_.rows() != sampling_points_.size() ||
            matrix_.cols() != basis->size()) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points_.size()) +
                                     "x" + std::to_string(basis->size()));
        }

        // Initialize SVD
        if (factorize) {
            matrix_svd_ = Eigen::JacobiSVD<Eigen::MatrixXd>(
                matrix_, Eigen::ComputeThinU | Eigen::ComputeThinV);
        }

        basis_ = basis;
    }

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

    Eigen::MatrixXd get_matrix() const { return matrix_; }

    const Eigen::VectorXd &sampling_points() const
    {
        return sampling_points_;
    }

    const Eigen::VectorXd &tau() const { return sampling_points_; }
    Eigen::JacobiSVD<Eigen::MatrixXd> get_matrix_svd() const
    {
        return matrix_svd_;
    }
};

inline Eigen::JacobiSVD<Eigen::MatrixXcd> make_split_svd(const Eigen::MatrixXcd &mat, bool has_zero = false)
{
    const int m = mat.rows(); // Number of rows in the input complex matrix
    const int n = mat.cols(); // Number of columns in the input complex matrix

    // Determine the starting row for the imaginary part.
    // If has_zero is true, skip the first row (offset = 1); otherwise, start at
    // row 0.
    const int offset_imag = has_zero ? 1 : 0;

    // Calculate the number of rows to be used from the imaginary part.
    const int imag_rows = m - offset_imag;

    // Total number of rows in the resulting real matrix:
    // all rows from the real part plus the selected rows from the imaginary
    // part.
    const int total_rows = m + imag_rows;

    // Create a real matrix with 'total_rows' rows and 'n' columns.
    Eigen::MatrixXcd rmat(total_rows, n);

    // Top part: assign the real part of the input matrix.
    rmat.topRows(m) = mat.real();

    // Bottom part: assign the selected block of the imaginary part.
    rmat.bottomRows(imag_rows) = mat.imag().block(offset_imag, 0, imag_rows, n);

    // Compute the SVD of the real matrix.
    // The options 'ComputeThinU' and 'ComputeThinV' compute the thin
    // (economical) versions of U and V.
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(rmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd;
}

/*
function MatsubaraSampling(basis::AbstractBasis; positive_only=false,
        sampling_points=default_matsubara_sampling_points(basis;
            positive_only), factorize=true)
    issorted(sampling_points) || sort!(sampling_points)
    if positive_only
        Int(first(sampling_points)) ≥ 0 || error("invalid negative sampling
frequencies") end matrix = eval_matrix(MatsubaraSampling, basis,
sampling_points) has_zero = iszero(first(sampling_points)) if factorize
        svd_matrix = positive_only ? SplitSVD(matrix; has_zero) : svd(matrix)
    else
        svd_matrix = nothing
    end
    sampling = MatsubaraSampling(sampling_points, matrix, svd_matrix,
positive_only) if factorize && iswellconditioned(basis) && cond(sampling) > 1e8
        @warn "Sampling matrix is poorly conditioned (cond =
$(cond(sampling)))." end return sampling end
*/
template <typename S>
class MatsubaraSampling : public AbstractSampling<S> {
private:
    std::shared_ptr<AbstractBasis<S>> basis_;
    std::vector<MatsubaraFreq<S>> sampling_points_;
    Eigen::MatrixXcd matrix_;
    Eigen::JacobiSVD<Eigen::MatrixXcd> matrix_svd_;
    bool positive_only_;
    bool has_zero_;

public:
    MatsubaraSampling(const std::shared_ptr<FiniteTempBasis<S>> &basis,
                       bool positive_only = false,
                       bool factorize = true)
        : positive_only_(positive_only)
    {
        // Get default sampling points from basis
        bool fence = false;
        sampling_points_ = default_matsubara_sampling_points(basis->uhat_full, basis->size(), fence, positive_only);
        std::sort(sampling_points_.begin(), sampling_points_.end());

        // Ensure matrix dimensions are correct
        if (sampling_points_.size() == 0) {
            throw std::runtime_error("No sampling points generated");
        }

        // Initialize evaluation matrix with correct dimensions
        matrix_ = eval_matrix(this, basis, sampling_points_);

        // Check matrix dimensions
        if (matrix_.rows() != sampling_points_.size() ||
            matrix_.cols() != basis->size()) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points_.size()) +
                                     "x" + std::to_string(basis->size()));
        }
        has_zero_ = sampling_points_[0].n == 0;
        // Initialize SVD
        if (factorize) {
            if (positive_only_) {
                matrix_svd_ = make_split_svd(matrix_, has_zero_);
            } else {
                matrix_svd_ = Eigen::JacobiSVD<Eigen::MatrixXcd>(
                    matrix_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            }
        }
        basis_ = basis;
    }

    MatsubaraSampling(const std::shared_ptr<AugmentedBasis<S>> &basis,
                       bool positive_only = false,
                       bool factorize = true)
        : basis_(basis), positive_only_(positive_only)
    {
        // Get default sampling points from basis
        bool fence = false;
        // Note that we use basis->basis->uhat_full, not basis->uhat_full
        sampling_points_ = default_matsubara_sampling_points(basis->basis->uhat_full, basis->size(), fence, positive_only);
        std::sort(sampling_points_.begin(), sampling_points_.end());

        // Ensure matrix dimensions are correct
        if (sampling_points_.size() == 0) {
            throw std::runtime_error("No sampling points generated");
        }

        // Initialize evaluation matrix with correct dimensions
        matrix_ = eval_matrix(this, basis, sampling_points_);

        // Check matrix dimensions
        if (matrix_.rows() != sampling_points_.size() ||
            matrix_.cols() != basis->size()) {
            throw std::runtime_error("Matrix dimensions mismatch: got " +
                                     std::to_string(matrix_.rows()) + "x" +
                                     std::to_string(matrix_.cols()) +
                                     ", expected " +
                                     std::to_string(sampling_points_.size()) +
                                     "x" + std::to_string(basis->size()));
        }
        has_zero_ = sampling_points_[0].n == 0;
        // Initialize SVD
        if (factorize) {
            if (positive_only_) {
                matrix_svd_ = make_split_svd(matrix_, has_zero_);
            } else {
                matrix_svd_ = Eigen::JacobiSVD<Eigen::MatrixXcd>(
                    matrix_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            }
        }
    }

    // Constructor that takes a DiscreteLehmannRepresentation and sampling points
    MatsubaraSampling(const DiscreteLehmannRepresentation<S> &dlr,
                      const std::vector<MatsubaraFreq<S>> &sampling_points,
                      bool positive_only = false,
                      bool factorize = true)
        : basis_(nullptr), sampling_points_(sampling_points), positive_only_(positive_only)
    {
        // Ensure matrix dimensions are correct
        if (sampling_points_.size() == 0) {
            throw std::runtime_error("No sampling points provided");
        }

        // Initialize evaluation matrix
        matrix_ = Eigen::MatrixXcd(sampling_points_.size(), dlr.size());

        // Fill the matrix with values from MatsubaraPoles
        for (size_t i = 0; i < sampling_points_.size(); ++i) {
            auto col = dlr.uhat(sampling_points_[i]);
            for (Eigen::Index j = 0; j < col.size(); ++j) {
                matrix_(i, j) = col(j);
            }
        }

        has_zero_ = sampling_points_[0].n == 0;

        // Initialize SVD
        if (factorize) {
            if (positive_only_) {
                matrix_svd_ = make_split_svd(matrix_, has_zero_);
            } else {
                matrix_svd_ = Eigen::JacobiSVD<Eigen::MatrixXcd>(
                    matrix_, Eigen::ComputeThinU | Eigen::ComputeThinV);
            }
        }
    }

    // Evaluate the basis coefficients at sampling points
    template <typename T, int N>
    Eigen::Tensor<std::complex<T>, N> evaluate(const Eigen::Tensor<T, N>& al, int dim = 0) const {
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

        // Create dimensions array for result tensor
        Eigen::array<Eigen::Index, N> dims;
        for (int i = 0; i < N; ++i) {
            dims[i] = (i == dim) ? matrix_.rows() : al.dimension(i);
        }

        // Create result tensor
        Eigen::Tensor<std::complex<T>, N> result(dims);

        // Convert input tensor to complex for contraction
        Eigen::Tensor<std::complex<T>, N> al_complex = al.template cast<std::complex<T>>();

        // Convert matrix to tensor
        Eigen::Tensor<std::complex<T>, 2> matrix_tensor(matrix_.rows(), matrix_.cols());
        for (Eigen::Index i = 0; i < matrix_.rows(); ++i) {
            for (Eigen::Index j = 0; j < matrix_.cols(); ++j) {
                matrix_tensor(i,j) = matrix_(i,j);
            }
        }

        // Specify contraction dimensions
        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
            Eigen::IndexPair<int>(1, dim)
        };

        // Perform contraction
        result = matrix_tensor.contract(al_complex, contract_dims);

        return result;
    }

    // Add these new overloads for Vector types
    template <typename T>
    Eigen::VectorX<std::complex<T>> evaluate(const Eigen::VectorX<T>& al) const {
        // Convert Vector to Tensor
        Eigen::Tensor<T, 1> al_tensor(al.size());
        for (Eigen::Index i = 0; i < al.size(); ++i) {
            al_tensor(i) = al(i);
        }

        // Use existing tensor-based evaluate
        auto result_tensor = evaluate(al_tensor, 0);

        // Convert result back to Vector
        Eigen::VectorX<std::complex<T>> result(result_tensor.dimension(0));
        for (Eigen::Index i = 0; i < result.size(); ++i) {
            result(i) = result_tensor(i);
        }
        return result;
    }

    // template <typename T, int N>
    // size_t workarrlength(const Eigen::Tensor<T, N> &ax, int dim) const
    //{
    // auto svd = get_matrix_svd();
    // return svd.singularValues().size() * (ax.size() / ax.dimension(dim));
    //}

    // Fit values at sampling points to basis coefficients
    template <typename T, int N>
    Eigen::Tensor<std::complex<T>, N> fit(const Eigen::Tensor<std::complex<T>, N> &ax, int dim = 0) const
    {
        if (dim < 0 || dim >= N) {
            throw std::runtime_error(
                "fit: dimension must be in [0..N). Got dim=" +
                std::to_string(dim));
        }
        auto svd = get_matrix_svd();
        if (positive_only_) {
            return fit_impl_split_svd<N>(svd, ax, dim, has_zero_);
        } else {
            return fit_impl<std::complex<T>, std::complex<T>, N>(svd, ax, dim);
        }
    }

    Eigen::MatrixXcd get_matrix() const { return matrix_; }

    const std::vector<MatsubaraFreq<S>> &sampling_points() const
    {
        return sampling_points_;
    }

    Eigen::JacobiSVD<Eigen::MatrixXcd> get_matrix_svd() const
    {
        return matrix_svd_;
    }
};

} // namespace sparseir
