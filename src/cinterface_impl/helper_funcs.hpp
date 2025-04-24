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
spir_sampling *
_spir_fermionic_tau_sampling_new(const spir_finite_temp_basis *b)
{
    std::shared_ptr<AbstractFiniteTempBasis> impl = get_impl_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    auto smpl =
        std::make_shared<sparseir::TauSampling<sparseir::Fermionic>>(*impl);
    return create_sampling(smpl);
}
