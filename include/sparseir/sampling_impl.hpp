#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/SVD>
#include <memory>
#include <stdexcept>
#include <vector>
#include <complex>
#include <tuple>

#include "sparseir/contraction.hpp"

namespace sparseir {

template <typename Scalar, typename InputScalar, typename OutputScalar>
void evaluate_inplace_dim2(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const Eigen::Map<const Eigen::Matrix<InputScalar, Eigen::Dynamic, Eigen::Dynamic>> &al,
    int dim,
    Eigen::Map<Eigen::Matrix<OutputScalar, Eigen::Dynamic, Eigen::Dynamic>> &output)
{
    // dim should be 0 or 1
    if (dim != 0 && dim != 1) {
        throw std::runtime_error("dim should be 0 or 1, but got " +
                                 std::to_string(dim));
    }
    if (dim == 0) {
        // (n_sampling_points, basis_size) * (basis_size, extra_size) =
        // (n_sampling_points, extra_size)
        _gemm_inplace(matrix.data(), al.data(), output.data(), matrix.rows(),
                      output.cols(), matrix.cols());
    } else {
        // (extra_size, basis_size) * (basis_size, n_sampling_points) =
        // (extra_size, n_sampling_points)
        _gemm_inplace_t(al.data(), matrix.data(), output.data(), al.rows(),
                        matrix.rows(), matrix.cols());
    }
}

// Common implementation for evaluate_inplace
template <typename Scalar, typename InputScalar, typename OutputScalar>
int evaluate_inplace_dim3(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &matrix,
    const Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> &input, int dim,
    Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> &output)
{
    using InputMatrix = Eigen::Matrix<InputScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using OutputMatrix = Eigen::Matrix<OutputScalar, Eigen::Dynamic, Eigen::Dynamic>;
    const int Dim = 3;

    if (dim < 0 || dim >= Dim) {
        // Invalid dimension
        return SPIR_INVALID_DIMENSION;
    }

    const auto basis_size = matrix.cols();
    const auto n_sampling_points = matrix.rows();

    if (basis_size != static_cast<std::size_t>(input.dimension(dim))) {
        // Dimension mismatch
        return SPIR_INPUT_DIMENSION_MISMATCH;
    }

    // extra dimension
    int extra_size = 1;
    for (int i = 0; i < Dim; i++) {
        if (i != dim) {
            extra_size *= input.dimension(i);
        }
    }

    if (dim == 0) {
        auto input_matrix =
            Eigen::Map<const InputMatrix>(input.data(), basis_size, extra_size);
        auto output_matrix = Eigen::Map<OutputMatrix>(
            output.data(), n_sampling_points, extra_size);
        evaluate_inplace_dim2(matrix, input_matrix, 0, output_matrix);
        return SPIR_COMPUTATION_SUCCESS;
    } else if (dim == Dim - 1) {
        auto input_matrix =
            Eigen::Map<const InputMatrix>(input.data(), extra_size, basis_size);
        auto output_matrix = Eigen::Map<OutputMatrix>(output.data(), extra_size,
                                                      n_sampling_points);
        evaluate_inplace_dim2(matrix, input_matrix, 1, output_matrix);
        return SPIR_COMPUTATION_SUCCESS;
    }

    // TODO: Cache buffers to avoid reallocation
    std::vector<InputScalar> input_buffer(basis_size * extra_size);

    auto input_dimensions = input.dimensions();

    // For dim == 1, we need to transpose (dim0, dim1, dim2) -> (dim1, dim0,
    // dim2) where dim1 is the basis dimension to be moved to first position
    auto input_transposed = Eigen::TensorMap<Eigen::Tensor<InputScalar, Dim>>(
        input_buffer.data(), basis_size, input_dimensions[0],
        input_dimensions[2]);

    auto output_transposed = Eigen::TensorMap<Eigen::Tensor<OutputScalar, Dim>>(
        output.data(), n_sampling_points, input_dimensions[0],
        input_dimensions[2]);

    // move the target dimension (dim=1) to the first position
    for (int k = 0; k < input_dimensions[2]; k++) {
        for (int j = 0; j < input_dimensions[1]; j++) {
            for (int i = 0; i < input_dimensions[0]; i++) {
                input_transposed(j, i, k) = input(i, j, k);
            }
        }
    }

    auto input_matrix =
        Eigen::Map<const InputMatrix>(&input_buffer[0], basis_size, extra_size);
    auto output_matrix =
        Eigen::Map<OutputMatrix>(output.data(), n_sampling_points, extra_size);
    evaluate_inplace_dim2(matrix, input_matrix, 0, output_matrix);

    // transpose back: (n_sampling_points, dim0, dim2) -> (dim0,
    // n_sampling_points, dim2)
    auto buffer = Eigen::Matrix<OutputScalar, Eigen::Dynamic, Eigen::Dynamic>(
        input_dimensions[0], n_sampling_points);
    for (int k = 0; k < input_dimensions[2]; k++) {
        for (int j = 0; j < input_dimensions[0]; j++) {
            for (int i = 0; i < n_sampling_points; i++) {
                buffer(j, i) = output_transposed(i, j, k);
            }
        }
        for (int i = 0; i < n_sampling_points; i++) {
            for (int j = 0; j < input_dimensions[0]; j++) {
                output(j, i, k) = buffer(j, i);
            }
        }
    }

    return SPIR_COMPUTATION_SUCCESS; // Success
}


template <typename Scalar, typename InputMatrixType, typename OutputMatrixType>
void fit_inplace_dim2(const sparseir::JacobiSVD<Eigen::MatrixX<Scalar>> &svd,
                    const InputMatrixType &input,
                    OutputMatrixType &output)
{
    using InputScalar = typename InputMatrixType::Scalar;
    using OutputScalar = typename OutputMatrixType::Scalar;

    // TODO: USE BLAS
    Eigen::Matrix<OutputScalar, Eigen::Dynamic, Eigen::Dynamic> UHB =
        svd.matrixU().adjoint() * input;

    // Apply inverse singular values to the rows of UHB
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        UHB.row(i) /= OutputScalar(svd.singularValues()(i));
    }
    _gemm_inplace(svd.matrixV().data(), UHB.data(), output.data(), svd.matrixV().rows(), output.cols(), svd.matrixV().cols());
}


template <typename Scalar, typename InputScalar, typename OutputScalar>
int fit_inplace_dim3(
    const sparseir::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> &svd,
    const Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> &input,
    int dim, Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> &output)
{
    using InputMatrix = Eigen::Matrix<InputScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using OutputMatrix = Eigen::Matrix<OutputScalar, Eigen::Dynamic, Eigen::Dynamic>;

    const int Dim = 3;
    const auto basis_size = svd.matrixV().rows();
    const auto n_sampling_points = svd.matrixU().rows();

    if (dim < 0 || dim >= Dim) {
        // Invalid dimension
        return SPIR_INVALID_DIMENSION;
    }

    if (n_sampling_points !=
        static_cast<int32_t>(input.dimension(dim))) {
        // Dimension mismatch
        return SPIR_INPUT_DIMENSION_MISMATCH;
    }

    auto input_dimensions = input.dimensions();
    auto output_dimensions = output.dimensions();
    output_dimensions[dim] = basis_size;

    // extra dimension
    int extra_size = 1;
    for (int i = 0; i < Dim; i++) {
        if (i != dim) {
            extra_size *= input_dimensions[i];
        }
    }

    // Move the target dimension to the first position
    Eigen::Tensor<InputScalar, 3> input_transposed = movedim(input, dim, 0);
    auto input_dimensions_transposed = input_transposed.dimensions();
    auto output_dimensions_transposed = input_dimensions_transposed;
    output_dimensions_transposed[0] = basis_size;
    Eigen::Tensor<OutputScalar, 3> output_transposed = Eigen::Tensor<OutputScalar, 3>(output_dimensions_transposed);
    
    // Calculate result using the existing fit method
    auto input_tranposed_matrix = Eigen::Map<const InputMatrix>(input_transposed.data(), n_sampling_points, extra_size);
    auto output_tranposed_matrix = Eigen::Map<OutputMatrix>(output_transposed.data(), basis_size, extra_size);
    fit_inplace_dim2(svd, input_tranposed_matrix, output_tranposed_matrix);

    // Transpose back: (basis_size, n_sampling_points, dim2) -> (n_sampling_points, basis_size, dim2)
    output = movedim(output_transposed, 0, dim);

    return SPIR_COMPUTATION_SUCCESS;
}


} // namespace sparseir