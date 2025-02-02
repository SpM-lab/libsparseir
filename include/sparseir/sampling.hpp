#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>
#include <functional>

namespace sparseir {

// Forward declarations
template <typename S>
class FiniteTempBasis;

template <typename S>
class TauSampling;

template <int N>
Eigen::array<int, N> getperm(int src, int dst) {
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

// movedim: テンソル arr の次元 src を次元 dst に移動する（他の次元の順序はそのまま）
template<typename T, int N>
Eigen::Tensor<T, N> movedim(const Eigen::Tensor<T, N>& arr, int src, int dst) {
    if (src == dst) {
        return arr;
    }
    auto perm = getperm<N>(src, dst);
    return arr.shuffle(perm);
}

// Helper function to calculate buffer size for tensors
template<typename T, int N>
std::vector<Eigen::Index> calculate_buffer_size(
    const Eigen::Tensor<T, N>& al,
    const Eigen::MatrixXd& matrix,
    int dim)
{
    std::vector<Eigen::Index> buffer_size;
    buffer_size.reserve(N);

    // Add dimensions up to dim
    for (int i = 0; i < dim; ++i) {
        buffer_size.push_back(al.dimension(i));
    }

    // Add matrix dimension
    buffer_size.push_back(matrix.rows());

    // Add remaining dimensions
    for (int i = dim + 1; i < N; ++i) {
        buffer_size.push_back(al.dimension(i));
    }

    return buffer_size;
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

// Define WorkSize struct
struct WorkSize {
    Eigen::Index rows;
    Eigen::Index cols;

    WorkSize(Eigen::Index r, Eigen::Index c) : rows(r), cols(c) {}

    Eigen::Index prod() const { return rows * cols; }
    Eigen::Index size() const { return rows; }
    Eigen::Index dimensions() const { return cols; }
};

// Forward declarations
template<typename T, int N>
void ldiv_noalloc_inplace(Eigen::VectorX<T>& flatbuffer,
                         const Eigen::BDCSVD<Eigen::MatrixX<T>>& svd,
                         const Eigen::VectorX<T>& flatarr,
                         Eigen::VectorX<T>& workarr);

template<typename T, int N>
void ldiv_noalloc(Eigen::VectorX<T>& buffer_perm,
                  const Eigen::BDCSVD<Eigen::MatrixX<T>>& svd,
                  const Eigen::VectorX<T>& arr_perm,
                  Eigen::VectorX<T>& workarr,
                  int offset);

template<typename T, int N>
void rdiv_noalloc_inplace(Eigen::VectorX<T>& flatbuffer,
                         const Eigen::VectorX<T>& flatarr,
                         const Eigen::BDCSVD<Eigen::MatrixX<T>>& svd,
                         Eigen::VectorX<T>& workarr);

// Helper function declarations
// Forward declarations
template <typename S>
class TauSampling;

template<typename S>
Eigen::VectorXd default_tau_sampling_points(const FiniteTempBasis<S>& basis) {
    double beta = basis.get_beta();
    int n_tau = basis.size() * 2;
    Eigen::VectorXd points(n_tau);
    for (int i = 0; i < n_tau; ++i) {
        points[i] = (i + 0.5) * beta / n_tau;
    }
    return points;
}

template<typename S>
Eigen::MatrixXd eval_matrix(const FiniteTempBasis<S>& basis,
                          const Eigen::VectorXd& x) {
    Eigen::MatrixXd matrix(x.size(), basis.size());
    for (int i = 0; i < x.size(); ++i) {
        auto u_i = basis.u(x[i]);
        for (int j = 0; j < basis.size(); ++j) {
            matrix(i, j) = u_i[j];
        }
    }
    return matrix;
}

// AbstractSampling クラスの定義
template<typename S>
class AbstractSampling {
public:
    virtual ~AbstractSampling() = default;
    virtual const Eigen::MatrixXd& matrix() const = 0;
    virtual const Eigen::JacobiSVD<Eigen::MatrixXd>& matrix_svd() const = 0;
    virtual const Eigen::VectorXd& sampling_points() const = 0;
};

template<typename S>
class TauSampling : public AbstractSampling<S> {
private:
    std::shared_ptr<FiniteTempBasis<S>> basis_;
    Eigen::VectorXd sampling_points_;
    Eigen::MatrixXd matrix_;
    Eigen::JacobiSVD<Eigen::MatrixXd> matrix_svd_;

public:
    explicit TauSampling(std::shared_ptr<FiniteTempBasis<S>> basis, bool factorize = true)
        : basis_(basis),
          sampling_points_(default_tau_sampling_points(*basis)),
          matrix_(eval_matrix(*basis, sampling_points_)) {
        if (factorize) {
            matrix_svd_.compute(matrix_);
        }
    }

    const Eigen::MatrixXd& matrix() const override { return matrix_; }
    const Eigen::JacobiSVD<Eigen::MatrixXd>& matrix_svd() const override { return matrix_svd_; }
    const Eigen::VectorXd& sampling_points() const override { return sampling_points_; }

    template<typename T>
    Eigen::Tensor<T, 4> evaluate(const Eigen::Tensor<T, 4>& gl, int dim) const {
        auto dims = gl.dimensions();
        Eigen::array<Eigen::Index, 4> new_dims;
        for (int i = 0; i < 4; ++i) {
            new_dims[i] = (i == dim) ? sampling_points_.size() : dims[i];
        }

        Eigen::Tensor<T, 4> result(new_dims);

        // Reshape for matrix multiplication
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> gl_mat(
            gl.data(), dims[dim], gl.size() / dims[dim]);

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result_mat =
            matrix_ * gl_mat;

        std::copy(result_mat.data(),
                 result_mat.data() + result_mat.size(),
                 result.data());

        return result;
    }
};

template <typename T, int N>
Eigen::MatrixX<T>& ldiv_alloc(
    Eigen::MatrixX<T>& Y,
    const Eigen::JacobiSVD<Eigen::MatrixXd>& A,
    const Eigen::MatrixX<T>& B,
    Eigen::VectorX<T>& workarr
) {
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
Eigen::Tensor<T, N>& div_noalloc_inplace(
    Eigen::Tensor<T, N>& buffer,
    const Eigen::JacobiSVD<Eigen::MatrixXd>& svd,
    const Eigen::Tensor<T, N>& arr,
    Eigen::VectorX<T>& workarr,
    int dim) {
    if (dim < 0 || dim >= N) {
        throw std::domain_error("Dimension must be in [0, N).");
    }
    if (dim == 0) {
        Eigen::Map<Eigen::MatrixX<T>> flatarr(arr.data(), arr.size(), 1);
        Eigen::Map<Eigen::MatrixX<T>> flatbuffer(buffer.data(), buffer.size(), 1);
        ldiv_noalloc_inplace<T, N>(flatbuffer, svd, flatarr, workarr);
        buffer = flatbuffer.reshaped(buffer.dimensions());
        return buffer;
    } else if (dim != N - 1) {
        auto perm = getperm<N>(dim, 0);
        Eigen::Tensor<T, N> arr_perm = arr.shuffle(perm).eval();
        Eigen::Tensor<T, N> buffer_perm = buffer.shuffle(perm).eval();
        ldiv_noalloc<T, N>(buffer_perm, svd, arr_perm, workarr, 0);
        auto inv_perm = getperm<N>(0, dim);
        buffer = buffer_perm.shuffle(inv_perm).eval();
        return buffer;
    } else {
        Eigen::Map<Eigen::MatrixX<T>> flatarr(arr.data(), arr.size(), 1);
        Eigen::Map<Eigen::MatrixX<T>> flatbuffer(buffer.data(), buffer.size(), 1);
        rdiv_noalloc_inplace<T, N>(flatbuffer, flatarr, svd, workarr);
        buffer = flatbuffer.reshaped(buffer.dimensions());
        return buffer;
    }
}

} // namespace sparseir

