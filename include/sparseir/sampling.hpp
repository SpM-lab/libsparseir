#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>
#include <functional>

namespace sparseir {

// Define WorkSize struct
struct WorkSize
{
    Eigen::Index rows;
    Eigen::Index cols;

    WorkSize(Eigen::Index r, Eigen::Index c) : rows(r), cols(c) { }

    Eigen::Index prod() const { return rows * cols; }
    Eigen::Index size() const { return rows; }
    Eigen::Index dimensions() const { return cols; }
};

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

// Forward declarations
template <typename T, int N>
Eigen::MatrixX<T>& ldiv_noalloc(
    Eigen::MatrixX<T>& Y,
    const Eigen::JacobiSVD<Eigen::MatrixXd>& A,
    const Eigen::MatrixX<T>& B,
    Eigen::VectorX<T>& workarr){
    /*
    function ldiv_noalloc!(Y::AbstractMatrix, A::SplitSVD, B::AbstractMatrix,
workarr) # Setup work space worksize = (size(A.UrealT, 1), size(B, 2))
worklength = prod(worksize)
length(workarr) ≥ worklength ||
    throw(DimensionMismatch("size(workarr)=$(size(workarr)), min
worksize=$worklength")) workarr_view = reshape(view(workarr, 1:worklength),
worksize)

mul!(workarr_view, A.UrealT, real(B))
mul!(workarr_view, A.UimagT, imag(B), true, true)
workarr_view ./= A.S
return mul!(Y, A.V, workarr_view)
end
*/
    WorkSize worksize(A.matrixU().cols(), B.cols());
    Eigen::Index worklength = worksize.prod();
    if (workarr.size() < worklength) {
        throw std::runtime_error("Work array is too small");
    }
    Eigen::Map<Eigen::MatrixX<T>> workarr_view(workarr.data(), worksize.rows, worksize.cols);
    workarr_view = A.matrixU().transpose() * B;
    workarr_view.array() /= A.singularValues().array();
    Y = A.matrixV() * workarr_view;
    return Y;
}

template <typename T, int N>
Eigen::MatrixX<T>& ldiv_noalloc_inplace(
    Eigen::MatrixX<T>& Y,
    const Eigen::JacobiSVD<Eigen::MatrixXd>& A,
    const Eigen::MatrixX<T>& B,
    Eigen::VectorX<T>& workarr){

    size_t worklength = A.matrixU().cols() * B.cols();

    if (workarr.size() < worklength) {
        throw std::runtime_error("Dimension mismatch: workarr.size()=" + std::to_string(workarr.size()) + ", min worksize=" + std::to_string(worklength));
    }
    Eigen::Map<Eigen::MatrixX<T>> workarr_view(workarr.data(), A.matrixU().cols(), B.cols());
    workarr_view = A.matrixU().transpose() * B;
    for (int i = 0; i < workarr_view.rows(); ++i) {
        for (int j = 0; j < workarr_view.cols(); ++j) {
            workarr_view(i, j) /= A.singularValues()(i);
        }
    }

    std::cout << "workarr_view.rows(): " << workarr_view.rows() << std::endl;
    std::cout << "workarr_view.cols(): " << workarr_view.cols() << std::endl;
    std::cout << "A.matrixV().rows(): " << A.matrixV().rows() << std::endl;
    std::cout << "A.matrixV().cols(): " << A.matrixV().cols() << std::endl;
    Y = A.matrixV() * workarr_view;
    return Y;
}

template <typename T, int N>
Eigen::MatrixX<T>& rdiv_noalloc_inplace(
    Eigen::MatrixX<T>& Y,
    const Eigen::MatrixX<T>& A,
    const Eigen::JacobiSVD<Eigen::MatrixXd>& B,
    Eigen::VectorX<T>& workarr){
    /*
    function rdiv_noalloc!(Y::AbstractMatrix, A::AbstractMatrix, B::SVD,
workarr) # Setup work space worksize = (size(A, 1), size(B.U, 2)) worklength =
prod(worksize) length(workarr) ≥ worklength ||
        throw(DimensionMismatch("size(workarr)=$(size(workarr)), min
worksize=$worklength")) workarr_view = reshape(view(workarr, 1:worklength),
worksize)

    # Note: conj creates a temporary matrix
    mul!(workarr_view, A, conj(B.U))
    workarr_view ./= reshape(B.S, 1, :)
    return mul!(Y, workarr_view, conj(B.Vt))
end
*/

    size_t worklength = A.rows() * B.matrixU().cols();
    if (workarr.size() < worklength) {
        throw std::runtime_error("Work array is too small");
    }
    Eigen::Map<Eigen::MatrixX<T>> workarr_view(workarr.data(), A.rows(), B.matrixU().cols());
    workarr_view = A * B.matrixU().conjugate();
    for (int i = 0; i < workarr_view.rows(); ++i) {
        for (int j = 0; j < workarr_view.cols(); ++j) {
            workarr_view(i, j) /= B.singularValues()(j);
        }
    }

    Y = workarr_view * B.matrixV().transpose();
    return Y;
}


// Helper function to calculate buffer size for tensors
template<typename T, int N>
Eigen::VectorX<Eigen::Index> calculate_buffer_size(
    const Eigen::Tensor<T, N>& al,
    const Eigen::MatrixXd& matrix,
    int dim)
{
    Eigen::VectorX<Eigen::Index> buffer_size(N);

    // Add dimensions up to dim
    for (int i = 0; i < dim; ++i) {
        buffer_size(i) = al.dimension(i);
    }

    // Add matrix dimension
    buffer_size(dim) = matrix.rows();

    // Add remaining dimensions
    for (int i = dim + 1; i < N; ++i) {
        buffer_size(i) = al.dimension(i);
    }

    return buffer_size;
}

template <typename T, int N>
inline Eigen::MatrixX<T> &
ldiv_alloc(Eigen::MatrixX<T> &Y, const Eigen::JacobiSVD<Eigen::MatrixXd> &A,
           const Eigen::MatrixX<T> &B, Eigen::VectorX<T> &workarr)
{
    WorkSize worksize(A.matrixU().cols(), B.cols());
    Eigen::Index worklength = worksize.prod();
    if (workarr.size() < worklength) {
        throw std::runtime_error("Work array is too small");
    }
    Eigen::Map<Eigen::MatrixX<T>> workarr_view(workarr.data(), worksize.rows,
                                               worksize.cols);
    workarr_view = A.matrixU().transpose() * B;
    workarr_view.array() /= A.singularValues().array();
    Y = A.matrixV() * workarr_view;
    return Y;
}

template <typename T, int N>
inline Eigen::Tensor<T, N>& div_noalloc_inplace(
    Eigen::Tensor<T, N>& buffer,
    const Eigen::JacobiSVD<Eigen::MatrixXd>& svd,
    const Eigen::Tensor<T, N>& arr,
    Eigen::VectorX<T>& workarr,
    int dim)
{
    if (dim < 0 || dim >= N) {
        throw std::domain_error("Dimension must be in [0, N).");
    }
    std::cout << "dim: " << dim << std::endl;
    if (dim == 0) {
        // Create a temporary matrix to hold the non-const data
        Eigen::MatrixX<T> flatarr = Eigen::Map<const Eigen::MatrixX<T>>(
            arr.data(), arr.dimension(0), arr.size() / arr.dimension(0));
        Eigen::MatrixX<T> flatbuffer(buffer.dimension(0), buffer.size() / buffer.dimension(0));
        std::cout << "Calling ldiv_noalloc_inplace" << std::endl;
        ldiv_noalloc_inplace<T, N>(flatbuffer, svd, flatarr, workarr);

        // Copy back to buffer tensor
        std::copy(flatbuffer.data(), flatbuffer.data() + flatbuffer.size(), buffer.data());
        return buffer;
    }
    else if (dim != N - 1) {
        auto perm = getperm<N>(dim, 0);
        Eigen::Tensor<T, N> arr_perm = movedim(arr, dim, 0);
        Eigen::Tensor<T, N> buffer_perm = movedim(buffer, dim, 0);

        Eigen::MatrixX<T> flatarr = Eigen::Map<const Eigen::MatrixX<T>>(
            arr_perm.data(), arr_perm.dimension(0), arr_perm.size() / arr_perm.dimension(0));
        Eigen::MatrixX<T> flatbuffer(buffer_perm.dimension(0), buffer_perm.size() / buffer_perm.dimension(0));

        // Remove the extra argument '0'
        ldiv_noalloc_inplace<T, N>(flatbuffer, svd, flatarr, workarr);
        std::copy(flatbuffer.data(), flatbuffer.data() + flatbuffer.size(), buffer_perm.data());
        auto inv_perm = getperm<N>(0, dim);
        buffer = buffer_perm.shuffle(inv_perm).eval();
        return buffer;
    }
    else {
        // Create temporary matrices for the last dimension case
        Eigen::MatrixX<T> flatarr = Eigen::Map<const Eigen::MatrixX<T>>(
            arr.data(), arr.size() / arr.dimension(N-1), arr.dimension(N-1));
        Eigen::MatrixX<T> flatbuffer(buffer.size() / buffer.dimension(N-1), buffer.dimension(N-1));

        rdiv_noalloc_inplace<T, N>(flatbuffer, flatarr, svd, workarr);

        // Copy back to buffer tensor
        std::copy(flatbuffer.data(), flatbuffer.data() + flatbuffer.size(), buffer.data());
        return buffer;
    }
}

/*
  A possible C++11/Eigen port of the Julia code that applies a matrix operation
  along a chosen dimension of an N-dimensional array.

  - matop_along_dim(buffer, mat, arr, dim, op) moves the requested dimension
    to the first or last as needed, then calls matop.
  - matop(buffer, mat, arr, op, dim) reshapes the tensor into a 2D view and
    calls the operation function.

  For simplicity, we assume:
    1) T is either float/double or a complex type.
    2) mat is an Eigen::MatrixXd (real) or possibly Eigen::MatrixXcd if needed.
    3) The user-provided op has the signature:
         void op(Eigen::Ref<Eigen::MatrixX<T>> out,
                 const Eigen::Ref<const Eigen::MatrixXd>& mat,
                 const Eigen::Ref<const Eigen::MatrixX<T>>& in);
       …or a variant suitable to your problem.
*/
template <typename T, int N>
Eigen::Tensor<T, N>& matop(
    Eigen::Tensor<T, N>& buffer,
    const Eigen::MatrixXd& mat,
    const Eigen::Tensor<T, N>& arr,
    int dim)
{
    if (dim != 0 && dim != N - 1) {
        throw std::domain_error("Dimension must be 0 or N-1 for matop.");
    }
    if (dim == 0) {
        Eigen::Index rowDim = arr.dimension(0);
        Eigen::Index colDim = arr.size() / rowDim;

        using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        if (mat.cols() != rowDim) {
            throw std::runtime_error("Matrix columns do not match input dimension");
        }

        Eigen::Map<const MatrixType> inMap(arr.data(), rowDim, colDim);
        Eigen::Map<MatrixType> outMap(buffer.data(), mat.rows(), colDim);

        outMap = mat * inMap;
        for (int i = 0; i < outMap.size(); ++i) {
            buffer.data()[i] = outMap.data()[i];
        }
    } else {
        Eigen::Index rowDim = arr.size() / arr.dimension(N - 1);
        Eigen::Index colDim = arr.dimension(N - 1);

        if (mat.cols() != colDim) {
            throw std::runtime_error("Matrix columns do not match input dimension");
        }

        using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        Eigen::Map<const MatrixType> inMap(arr.data(), rowDim, colDim);
        Eigen::Map<MatrixType> outMap(buffer.data(), rowDim, mat.rows());

        outMap = inMap * mat.transpose();

        for (int i = 0; i < outMap.size(); ++i) {
            buffer.data()[i] = outMap.data()[i];
        }
    }
    return buffer;
}

template <typename T, int N>
Eigen::Tensor<T, N>& matop_along_dim(
    Eigen::Tensor<T, N>& buffer,
    const Eigen::MatrixXd& mat,
    const Eigen::Tensor<T, N>& arr,
    int dim
) {
    if (dim < 0 || dim >= N) {
        throw std::domain_error("Dimension must be in [0, N).");
    }

    if (dim == 0) {
        return matop<T, N>(buffer, mat, arr, 0);
    } else if (dim != N - 1) {
        auto perm = getperm<N>(dim, 0);
        Eigen::Tensor<T, N> arr_perm = arr.shuffle(perm).eval();
        Eigen::Tensor<T, N> buffer_perm = buffer.shuffle(perm).eval();

        matop<T, N>(buffer_perm, mat, arr_perm, 0);

        auto inv_perm = getperm<N>(0, dim);
        buffer = buffer_perm.shuffle(inv_perm).eval();
    } else {
        return matop<T, N>(buffer, mat, arr, N - 1);
    }

    return buffer;
}

template <typename S>
class AbstractSampling {
public:
    virtual ~AbstractSampling() = default;

    // Evaluate the basis coefficients at sampling points
    template <typename T, int N>
    Eigen::Tensor<T, N> evaluate(const Eigen::Tensor<T, N> &al,
                                 int dim = 1) const
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

        // Calculate buffer dimensions using the new tensor version
        auto buffer_dims = calculate_buffer_size(al, get_matrix(), dim);

        // Convert vector to array for tensor construction
        Eigen::array<Eigen::Index, N> dims;
        std::copy(buffer_dims.begin(), buffer_dims.end(), dims.begin());

        // Create buffer with calculated dimensions
        Eigen::Tensor<T, N> buffer(dims);

        matop_along_dim<T, N>(buffer, get_matrix(), al, dim);
        return buffer;
    }

    template <typename T, int N>
    size_t workarrlength(const Eigen::Tensor<T, N> &ax, int dim) const
    {
        auto svd = get_matrix_svd();
        return svd.singularValues().size() * (ax.size() / ax.dimension(dim));
    }

    // Fit values at sampling points to basis coefficients
    template <typename T, int N>
    Eigen::Tensor<T, N> fit(const Eigen::Tensor<T, N> &ax,
                            int dim = 1) const
    {
        if (dim < 0 || dim >= N) {
            throw std::runtime_error(
                "fit: dimension must be in [0..N). Got dim=" +
                std::to_string(dim));
        }
        Eigen::VectorX<T> workarr(workarrlength(ax, dim));
        auto buffer_dims = calculate_buffer_size(ax, get_matrix(), dim);
        Eigen::array<Eigen::Index, N> dims;
        std::copy(buffer_dims.begin(), buffer_dims.end(), dims.begin());
        Eigen::Tensor<T, N> buffer(dims);
        // std::cout << "buffer.dimensions(): " << buffer.dimensions() << std::endl;
        auto svd = get_matrix_svd();
        // std::cout << "svd.matrixU().cols(): " << svd.matrixU().cols() << std::endl;
        // std::cout << "svd.matrixV().rows(): " << svd.matrixV().rows() << std::endl;
        // std::cout << "svd.singularValues().size(): " << svd.singularValues().size() << std::endl;
        // std::cout << "ax.dimension(dim): " << ax.dimension(dim) << std::endl;

        if (workarr.size() < workarrlength(ax, dim)) {
            throw std::runtime_error("Work array is too small");
        }
        div_noalloc_inplace<T, N>(buffer, svd, ax, workarr, dim);
        return buffer;
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

