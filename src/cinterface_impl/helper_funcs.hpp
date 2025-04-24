#include <memory>
#include <stdexcept>
#include <cstdint>
#include <iostream>

// Helper function to convert N-dimensional array to 3D array by collapsing
// dimensions
static std::array<int32_t, 3> collapse_to_3d(int32_t ndim, const int32_t *dims,
                                             int32_t target_dim)
{
    std::array<int32_t, 3> dims_3d = {1, dims[target_dim], 1};
    // Multiply all dimensions before target_dim into first dimension
    for (int32_t i = 0; i < target_dim; ++i) {
        dims_3d[0] *= dims[i];
    }
    // Multiply all dimensions after target_dim into last dimension
    for (int32_t i = target_dim + 1; i < ndim; ++i) {
        dims_3d[2] *= dims[i];
    }
    return dims_3d;
}

// Helper function to convert N-dimensional array to 2D array by collapsing
// dimensions
static std::array<int32_t, 2> collapse_to_2d(int32_t ndim, const int32_t *dims,
                                             int32_t target_dim)
{
    std::array<int32_t, 2> dims_2d = {dims[target_dim], 1};
    // Multiply all dimensions before target_dim into first dimension
    for (int32_t i = 0; i < target_dim; ++i) {
        dims_2d[0] *= dims[i];
    }
    // Multiply all dimensions after target_dim into last dimension
    for (int32_t i = target_dim + 1; i < ndim; ++i) {
        dims_2d[1] *= dims[i];
    }
    return dims_2d;
}

// Template function to handle all evaluation cases - moved outside extern "C"
// block
template <typename InputScalar, typename OutputScalar>
static int
evaluate_impl(const spir_sampling *s, spir_order_type order, int32_t ndim,
              int32_t *input_dims, int32_t target_dim, const InputScalar *input,
              OutputScalar *out,
              int (sparseir::AbstractSampling::*eval_func)(
                  const Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> &,
                  int, Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> &)
                  const)
{
    auto impl = get_impl_sampling(s);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    // Convert dimensions
    std::array<int32_t, 3> dims_3d =
        collapse_to_3d(ndim, input_dims, target_dim);

    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());
        std::array<int32_t, 3> output_dims_3d = input_dims_3d;

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);
        // Convert to column-major order for Eigen
        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    } else {
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::array<int32_t, 3> output_dims_3d = input_dims_3d;
        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);

        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    }
}

template <typename InputScalar, typename OutputScalar>
static int
fit_impl(const spir_sampling *s, spir_order_type order, int32_t ndim,
         int32_t *input_dims, int32_t target_dim, const InputScalar *input,
         OutputScalar *out,
         int (sparseir::AbstractSampling::*eval_func)(
             const Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> &, int,
             Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> &) const)
{
    auto impl = get_impl_sampling(s);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    // Convert dimensions
    std::array<int32_t, 3> dims_3d =
        collapse_to_3d(ndim, input_dims, target_dim);

    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());

        std::array<int32_t, 3> output_dims_3d = input_dims_3d;

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);
        // Convert to column-major order for Eigen
        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    } else {
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::array<int32_t, 3> output_dims_3d = input_dims_3d;

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);

        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    }
}

template<typename InternalType>
spir_singular_funcs* _create_singular_funcs(std::shared_ptr<InternalType> impl) {
    return create_singular_funcs(
        std::static_pointer_cast<AbstractContinuousFunctions>(
            std::make_shared<ContinuousFunctions<InternalType>>(impl)
        )
    );
}


template<typename S>
int32_t spir_dlr_to_IR(const spir_dlr *dlr, spir_order_type order,
                           int32_t ndim, int32_t *input_dims,
                           const double *input, double *out)
{
    std::shared_ptr<DLR<S>> impl = std::dynamic_pointer_cast<DLR<S>>(get_impl_dlr(dlr));
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    std::array<int32_t, 2> input_dims_2d = collapse_to_2d(ndim, input_dims, 0);
    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::reverse(input_dims_2d.begin(), input_dims_2d.end());
    }
    Eigen::Tensor<double, 2> input_tensor(input_dims_2d[0], input_dims_2d[1]);
    size_t total_input_size = input_dims_2d[0] * input_dims_2d[1];
    for (size_t i = 0; i < total_input_size; i++) {
        input_tensor.data()[i] = input[i];
    }
    Eigen::Tensor<double, 2> out_tensor = impl->get_impl()->to_IR(input_tensor);
    size_t total_output_size =
        out_tensor.dimension(0) * out_tensor.dimension(1);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor.data()[i];
    }
    return SPIR_COMPUTATION_SUCCESS;
}


template<typename S>
int32_t spir_dlr_from_IR(const spir_dlr *dlr, spir_order_type order,
                             int32_t ndim, int32_t *input_dims,
                             const double *input, double *out)
{
    std::shared_ptr<DLR<S>> impl = std::dynamic_pointer_cast<DLR<S>>(get_impl_dlr(dlr));
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    std::array<int32_t, 2> input_dims_2d = collapse_to_2d(ndim, input_dims, 0);
    Eigen::Tensor<double, 2> input_tensor(input_dims_2d[0], input_dims_2d[1]);
    std::size_t total_input_size = input_dims_2d[0] * input_dims_2d[1];
    for (std::size_t i = 0; i < total_input_size; i++) {
        input_tensor.data()[i] = input[i];
    }

    Eigen::Tensor<double, 2> out_tensor = impl->get_impl()->from_IR(input_tensor);

    // pass data to out
    std::size_t total_output_size =
        out_tensor.dimension(0) * out_tensor.dimension(1);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor.data()[i];
    }
    return SPIR_COMPUTATION_SUCCESS;
}
