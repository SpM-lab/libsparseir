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

    // output ndim, target_dim, dims_3d
    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());
        std::array<int32_t, 3> output_dims_3d = input_dims_3d;
        output_dims_3d[1] = impl.get()->n_sampling_points();

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);
        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    } else {
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::array<int32_t, 3> output_dims_3d = input_dims_3d;
        output_dims_3d[1] = impl.get()->n_sampling_points();

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
    if (!impl.get())
        return SPIR_GET_IMPL_FAILED;

    // Convert dimensions
    std::array<int32_t, 3> dims_3d =
        collapse_to_3d(ndim, input_dims, target_dim);

    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::array<int32_t, 3> input_dims_3d = dims_3d;
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());

        std::array<int32_t, 3> output_dims_3d = input_dims_3d;
        output_dims_3d[1] = impl.get()->basis_size();

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
        output_dims_3d[1] = impl.get()->basis_size();

        // Create TensorMaps
        Eigen::TensorMap<const Eigen::Tensor<InputScalar, 3>> input_3d(
            input, input_dims_3d);
        Eigen::TensorMap<Eigen::Tensor<OutputScalar, 3>> output_3d(
            out, output_dims_3d);

        return (impl.get()->*eval_func)(input_3d, 1, output_3d);
    }
}

template <typename S>
spir_funcs *_create_ir_tau_funcs(std::shared_ptr<sparseir::TauFunctions<S,sparseir::PiecewiseLegendrePoly>> impl, double beta)
{
    return create_funcs(std::static_pointer_cast<AbstractContinuousFunctions>(
        std::make_shared<TauFunctions<sparseir::TauFunctions<S,sparseir::PiecewiseLegendrePoly>>>(impl, beta)));
}

//template <typename S>
//spir_funcs *_create_dlr_tau_funcs(std::shared_ptr<sparseir::TauFunctions<S,sparseir::TauPoles<S>>> impl, double beta)
//{
    //return create_funcs(std::static_pointer_cast<AbstractContinuousFunctions>(
        //std::make_shared<TauFunctions<sparseir::TauFunctions<S,sparseir::TauPoles<S>>>>(impl, beta)));
//}

template <typename InternalType>
spir_funcs *_create_omega_funcs(std::shared_ptr<InternalType> impl)
{
    return create_funcs(std::static_pointer_cast<AbstractContinuousFunctions>(
        std::make_shared<OmegaFunctions<InternalType>>(impl)));
}

template <typename S, typename T>
int32_t spir_dlr_to_IR(const spir_dlr *dlr, spir_order_type order, int32_t ndim,
                       int32_t *input_dims, int32_t target_dim, const T *input, T *out)
{
    std::shared_ptr<_DLR<S>> impl =
        std::dynamic_pointer_cast<_DLR<S>>(get_impl_dlr(dlr));
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    std::array<int32_t, 3> input_dims_3d = collapse_to_3d(ndim, input_dims, target_dim);
    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());
    }
    Eigen::Tensor<T, 3> input_tensor(input_dims_3d[0], input_dims_3d[1], input_dims_3d[2]);
    size_t total_input_size = input_dims_3d[0] * input_dims_3d[1] * input_dims_3d[2];
    for (size_t i = 0; i < total_input_size; i++) {
        input_tensor.data()[i] = input[i];
    }
    Eigen::Tensor<T, 3> out_tensor = impl->get_impl()->to_IR(input_tensor, 1);
    size_t total_output_size =
        out_tensor.dimension(0) * out_tensor.dimension(1) * out_tensor.dimension(2);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor.data()[i];
    }
    return SPIR_COMPUTATION_SUCCESS;
}

template <typename S, typename T>
int32_t spir_dlr_from_IR(const spir_dlr *dlr, spir_order_type order,
                         int32_t ndim, int32_t *input_dims, int32_t target_dim,
                         const T *input, T *out)
{
    std::shared_ptr<_DLR<S>> impl =
        std::dynamic_pointer_cast<_DLR<S>>(get_impl_dlr(dlr));
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    std::array<int32_t, 3> input_dims_3d = collapse_to_3d(ndim, input_dims, target_dim);
    if (order == SPIR_ORDER_ROW_MAJOR) {
        std::reverse(input_dims_3d.begin(), input_dims_3d.end());
    }

    Eigen::Tensor<T, 3> input_tensor_3d(input_dims_3d[0], input_dims_3d[1], input_dims_3d[2]);
    std::size_t total_input_size = input_dims_3d[0] * input_dims_3d[1] * input_dims_3d[2];
    for (std::size_t i = 0; i < total_input_size; i++) {
        input_tensor_3d.data()[i] = input[i];
    }
    // move the target dimension to the first dimension
    input_tensor_3d = sparseir::movedim(input_tensor_3d, 1, 0);
    // reshape to 2D
    Eigen::array<Eigen::Index, 2> input2d_dims{input_dims_3d[1], input_dims_3d[0] * input_dims_3d[2]};
    Eigen::Tensor<T, 2> input_tensor_2d = input_tensor_3d.reshape(input2d_dims);
    Eigen::Tensor<T, 2> out_tensor_2d = impl->get_impl()->from_IR(input_tensor_2d);
    // move the target dimension to the last dimension
    // reshape to 3D
    Eigen::array<Eigen::Index, 3> out3d_dims{out_tensor_2d.dimension(0), input_dims_3d[0], input_dims_3d[2]};

    Eigen::Tensor<T, 3> out_tensor_3d_ = out_tensor_2d.reshape(out3d_dims);
    Eigen::Tensor<T, 3> out_tensor_3d = sparseir::movedim(out_tensor_3d_, 0, 1);

    // pass data to out
    std::size_t total_output_size =
        out_tensor_3d.dimension(0) * out_tensor_3d.dimension(1) * out_tensor_3d.dimension(2);
    for (std::size_t i = 0; i < total_output_size; i++) {
        out[i] = out_tensor_3d.data()[i];
    }
    return SPIR_COMPUTATION_SUCCESS;
}

template <typename S, typename SMPL>
spir_sampling *_spir_sampling_new(const spir_finite_temp_basis *b)
{
    std::shared_ptr<AbstractFiniteTempBasis> impl =
        get_impl_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    auto impl_finite_temp_basis =
        std::static_pointer_cast<_FiniteTempBasis<S>>(impl)->get_impl();
    return create_sampling(std::static_pointer_cast<sparseir::AbstractSampling>(
        std::make_shared<SMPL>(impl_finite_temp_basis)));
}


template <typename S, typename SMPL>
spir_sampling *_spir_matsubara_sampling_new(const spir_finite_temp_basis *b, bool positive_only)
{
    std::shared_ptr<AbstractFiniteTempBasis> impl =
        get_impl_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    auto impl_finite_temp_basis =
        std::static_pointer_cast<_FiniteTempBasis<S>>(impl)->get_impl();
    return create_sampling(std::static_pointer_cast<sparseir::AbstractSampling>(
        std::make_shared<SMPL>(impl_finite_temp_basis, positive_only)));
}

template <typename S, typename SMPL>
spir_sampling *_spir_matsubara_sampling_dlr_new(const spir_dlr *dlr, int32_t n_smpl_points, const int32_t *smpl_points, bool positive_only)
{
    auto impl = get_impl_dlr(dlr);
    if (!impl)
        return nullptr;

    std::vector<sparseir::MatsubaraFreq<S>> smpl_points_vec;
    smpl_points_vec.reserve(n_smpl_points);
    for (int32_t i = 0; i < n_smpl_points; ++i) {
        smpl_points_vec.emplace_back(smpl_points[i]);
    }

    auto dlr_impl = std::static_pointer_cast<_DLR<S>>(impl)->get_impl();
    auto sampling = std::make_shared<SMPL>(dlr_impl, smpl_points_vec, positive_only);
    return create_sampling(sampling);
}



template <typename S>
spir_dlr *_spir_dlr_new(const spir_finite_temp_basis *b)
{
    auto impl = get_impl_finite_temp_basis(b);
    if (!impl) {
        return nullptr;
    }

    auto ptr_finite_temp_basis =
        std::static_pointer_cast<_FiniteTempBasis<S>>(impl)->get_impl();
    auto dlr = std::make_shared<sparseir::DiscreteLehmannRepresentation<S>>(*ptr_finite_temp_basis);

    auto ptr_dlr = std::make_shared<_DLR<S>>(
        std::make_shared<sparseir::DiscreteLehmannRepresentation<S>>(
            *ptr_finite_temp_basis));
    return create_dlr(std::static_pointer_cast<AbstractDLR>(ptr_dlr));
}

template <typename S>
spir_dlr *_spir_dlr_new_with_poles(const spir_finite_temp_basis *b,
                                   const int npoles, const double *poles)
{
    auto impl = get_impl_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    auto ptr_finite_temp_basis =
        std::static_pointer_cast<_FiniteTempBasis<S>>(impl)->get_impl();

    Eigen::VectorXd poles_vec(npoles);
    for (int i = 0; i < npoles; i++) {
        poles_vec(i) = poles[i];
    }

    auto ptr_dlr = std::make_shared<_DLR<S>>(
        std::make_shared<sparseir::DiscreteLehmannRepresentation<S>>(
            *ptr_finite_temp_basis, poles_vec));
    return create_dlr(std::static_pointer_cast<AbstractDLR>(ptr_dlr));
}

template <typename S>
int32_t _spir_dlr_get_u(const spir_dlr *dlr, spir_funcs **u)
{
    try {
        std::shared_ptr<AbstractDLR> impl = get_impl_dlr(dlr);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        auto beta = impl->get_beta();
        if (beta <= 0) {
            throw std::runtime_error("beta is less than or equal to 0");
        }

        std::shared_ptr<AbstractContinuousFunctions> u_funcs = std::static_pointer_cast<_DLR<S>>(impl)->get_u();
        *u = create_funcs(u_funcs);
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

template <typename S>
int32_t _spir_dlr_get_uhat(const spir_dlr *dlr, spir_matsubara_funcs **uhat)
{
    try {
        auto impl = get_impl_dlr(dlr);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        *uhat = create_matsubara_funcs(impl->get_uhat());
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

template <typename S>
int32_t _spir_finite_temp_basis_get_u(const spir_finite_temp_basis *b,
                                      spir_funcs **u)
{
    try {
        std::shared_ptr<AbstractFiniteTempBasis> impl = get_impl_finite_temp_basis(b);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        std::shared_ptr<sparseir::TauFunctions<S, sparseir::PiecewiseLegendrePoly>> u_impl = std::static_pointer_cast<_FiniteTempBasis<S>>(impl)->get_impl()->u;
        *u = _create_ir_tau_funcs<S>(u_impl, impl->get_beta());
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

template <typename S>
int32_t _spir_finite_temp_basis_get_v(const spir_finite_temp_basis *b,
                                      spir_funcs **v)
{
    try {
        auto impl = get_impl_finite_temp_basis(b);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        *v = _create_omega_funcs(impl->get_v());
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

template <typename S>
int32_t _spir_finite_temp_basis_get_uhat(const spir_finite_temp_basis *b,
                                         spir_matsubara_funcs **uhat)
{
    try {
        auto impl = get_impl_finite_temp_basis(b);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        *uhat = create_matsubara_funcs(impl->get_uhat());
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

template <typename S>
int32_t _spir_matsubara_funcs_get_size(const spir_matsubara_funcs* funcs, int32_t* size) {
    try {
        auto impl = get_impl_matsubara_funcs(funcs);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        *size = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception& e) {
        return SPIR_GET_IMPL_FAILED;
    }
}


template <typename K>
spir_finite_temp_basis* _spir_finite_temp_basis_new_with_sve(
    spir_statistics_type statistics, double beta, double omega_max,
    const K& kernel, const spir_sve_result *sve)
{
    try {
        auto sve_impl = get_impl_sve_result(sve);
        if (!sve_impl)
            return nullptr;
        if (statistics == SPIR_STATISTICS_FERMIONIC) {
            using FiniteTempBasisType =
                sparseir::FiniteTempBasis<sparseir::Fermionic>;
            auto impl = std::make_shared<FiniteTempBasisType>(
                beta, omega_max, kernel, *sve_impl);
            return create_finite_temp_basis(
                std::make_shared<_FiniteTempBasis<sparseir::Fermionic>>(impl));
        } else {
            using FiniteTempBasisType =
                sparseir::FiniteTempBasis<sparseir::Bosonic>;
            auto impl = std::make_shared<FiniteTempBasisType>(
                beta, omega_max, kernel, *sve_impl);
            return create_finite_temp_basis(
                std::make_shared<_FiniteTempBasis<sparseir::Bosonic>>(impl));
        }
    } catch (...) {
        return nullptr;
    }
}
