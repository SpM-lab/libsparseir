#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>
#include <array>
#include <functional>
#include <numeric>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <sparseir/sparseir.hpp> // C++ interface
#include <sparseir/sparseir.h>   // C interface

using Catch::Approx;

// assert working in non-debug mode
inline void _assert(bool cond)
{
    if (!cond) {
        std::cerr << "Assertion failed" << std::endl;
        abort();
    }
}

template <typename S>
spir_statistics_type get_stat()
{
    if (std::is_same<S, sparseir::Fermionic>::value) {
        return SPIR_STATISTICS_FERMIONIC;
    } else {
        return SPIR_STATISTICS_BOSONIC;
    }
}

// spir_order_type get_order(Eigen::StorageOptions order)
//{
// if (order == Eigen::ColMajor) {
// return SPIR_ORDER_COLUMN_MAJOR;
//} else {
// return SPIR_ORDER_ROW_MAJOR;
//}
//}

template <int ndim>
std::array<Eigen::Index, ndim> _get_dims(int target_dim_size,
                                         const std::vector<int> &extra_dims,
                                         int target_dim)
{
    std::array<Eigen::Index, ndim> dims;
    dims[target_dim] = target_dim_size;
    int pos = 0;
    for (int i = 0; i < extra_dims.size(); ++i) {
        if (i == target_dim) {
            continue;
        }
        dims[pos] = extra_dims[i];
        ++pos;
    }
    return dims;
}

// Helper function to evaluate basis functions at multiple points
template <typename T, Eigen::StorageOptions ORDER>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER>
_evaluate_basis_functions(const spir_funcs* u, const Eigen::VectorXd& x_values) {
    int32_t status;
    int32_t funcs_size;
    status = spir_funcs_get_size(u, &funcs_size);
    _assert(status == SPIR_COMPUTATION_SUCCESS);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER> u_eval_mat(
        x_values.size(), funcs_size);
    for (Eigen::Index i = 0; i < x_values.size(); ++i) {
        Eigen::VectorXd u_eval(funcs_size);
        status = spir_evaluate_funcs(u, x_values(i), u_eval.data());
        _assert(status == SPIR_COMPUTATION_SUCCESS);
        u_eval_mat.row(i) = u_eval.transpose().cast<T>();
    }
    return u_eval_mat;
}

// Helper function to evaluate Matsubara basis functions at multiple frequencies
template <typename T, Eigen::StorageOptions ORDER>
Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, ORDER>
_evaluate_matsubara_basis_functions(const spir_matsubara_funcs* uhat, 
                                  const Eigen::VectorXi& matsubara_indices) {
    int32_t status;
    int32_t funcs_size;
    status = spir_matsubara_funcs_get_size(uhat, &funcs_size);
    _assert(status == SPIR_COMPUTATION_SUCCESS);

    Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, ORDER> uhat_eval_mat(
        matsubara_indices.size(), funcs_size);
    
    std::vector<std::complex<T>> uhat_eval(funcs_size);
    for (Eigen::Index i = 0; i < matsubara_indices.size(); ++i) {
        int32_t freq = matsubara_indices(i);
        status = spir_evaluate_matsubara_funcs(uhat, 
                                             ORDER == Eigen::ColMajor ? SPIR_ORDER_COLUMN_MAJOR : SPIR_ORDER_ROW_MAJOR,
                                             1, 
                                             &freq, 
                                             reinterpret_cast<c_complex*>(uhat_eval.data()));
        _assert(status == SPIR_COMPUTATION_SUCCESS);
        uhat_eval_mat.row(i) = Eigen::Map<Eigen::VectorX<std::complex<T>>>(uhat_eval.data(), funcs_size);
    }
    return uhat_eval_mat;
}

// Helper function to perform the tensor transformation
template <typename T, int ndim, Eigen::StorageOptions ORDER, typename BasisEvalType>
Eigen::Tensor<typename BasisEvalType::Scalar, ndim, ORDER>
_transform_coefficients(const Eigen::Tensor<T, ndim, ORDER>& coeffs,
                       const BasisEvalType& basis_eval,
                       int target_dim) {
    Eigen::Tensor<T, ndim, ORDER> coeffs_targetdim0 =
        sparseir::movedim(coeffs, target_dim, 0);

    // Calculate extra dimensions size
    Eigen::Index extra_size = 1;
    for (int i = 1; i < ndim; ++i) {
        extra_size *= coeffs_targetdim0.dimension(i);
    }

    // Create result tensor
    std::array<Eigen::Index, ndim> dims;
    dims[0] = basis_eval.rows();
    for (int i = 1; i < ndim; ++i) {
        dims[i] = coeffs_targetdim0.dimension(i);
    }
    Eigen::Tensor<typename BasisEvalType::Scalar, ndim, ORDER> result(dims);

    // Map input tensors to matrices
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER>>
        coeffs_mat(coeffs_targetdim0.data(), coeffs_targetdim0.dimension(0), extra_size);
    Eigen::Map<Eigen::Matrix<typename BasisEvalType::Scalar, Eigen::Dynamic, Eigen::Dynamic, ORDER>> result_mat(
        result.data(), basis_eval.rows(), extra_size);

    // Perform single matrix multiplication
    result_mat = basis_eval * coeffs_mat;

    // Move dimensions back to original order
    return sparseir::movedim(result, 0, target_dim);
}

template <typename T, int ndim, Eigen::StorageOptions ORDER>
Eigen::Tensor<T, ndim, ORDER>
_evaluate_gtau(const Eigen::Tensor<T, ndim, ORDER>& coeffs,
              const spir_funcs* u, int target_dim,
              const Eigen::VectorXd& x_values) {
    auto u_eval_mat = _evaluate_basis_functions<T, ORDER>(u, x_values);
    return _transform_coefficients<T, ndim, ORDER>(coeffs, u_eval_mat, target_dim);
}

template <typename T, int ndim, Eigen::StorageOptions ORDER>
Eigen::Tensor<std::complex<T>, ndim, ORDER>
_evaluate_giw(const Eigen::Tensor<T, ndim, ORDER>& coeffs,
             const spir_matsubara_funcs* uhat, int target_dim,
             const Eigen::VectorXi& matsubara_indices) {
    auto uhat_eval_mat = _evaluate_matsubara_basis_functions<T, ORDER>(uhat, matsubara_indices);
    return _transform_coefficients<T, ndim, ORDER>(coeffs, uhat_eval_mat, target_dim);
}

template <typename T, int ndim, Eigen::StorageOptions ORDER>
bool compare_tensors_with_relative_error(const Eigen::Tensor<T, ndim, ORDER> &a,
                                         const Eigen::Tensor<T, ndim, ORDER> &b,
                                         double epsilon)
{
    Eigen::Tensor<T, ndim, ORDER> diff = (a - b).abs();

    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> diff_vec(diff.data(),
                                                                   diff.size());
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> ref_vec(a.data(),
                                                                  a.size());
    T max_diff = diff_vec.maxCoeff();
    T max_ref = ref_vec.maxCoeff();

    return max_diff <= epsilon * max_ref;
}

template <typename S, typename K, int ndim, Eigen::StorageOptions ORDER>
void integration_test(double beta, double wmax, double epsilon,
                      const std::vector<int> &extra_dims, int target_dim,
                      const spir_order_type order)
{
    if (target_dim != 0) {
        std::cerr << "target_dim must be 0" << std::endl;
        return;
    }
    if (ndim != 1 + extra_dims.size()) {
        std::cerr << "ndim must be 1 + extra_dims.size()" << std::endl;
    }

    // Verify that the template parameter matches the runtime parameter
    if (ORDER == Eigen::ColMajor) {
        REQUIRE(order == SPIR_ORDER_COLUMN_MAJOR);
    } else {
        REQUIRE(order == SPIR_ORDER_ROW_MAJOR);
    }

    auto stat = get_stat<S>();
    int32_t status;

    // IR basis
    spir_finite_temp_basis *basis =
        spir_finite_temp_basis_new(stat, beta, wmax, epsilon);
    REQUIRE(basis != nullptr);
    int32_t basis_size;
    status = spir_finite_temp_basis_get_size(basis, &basis_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Tau Sampling
    spir_sampling *tau_sampling = spir_tau_sampling_new(basis);
    REQUIRE(tau_sampling != nullptr);
    int32_t num_tau_points;
    status = spir_sampling_get_num_points(tau_sampling, &num_tau_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    Eigen::VectorXd tau_points(num_tau_points);
    status = spir_sampling_get_tau_points(tau_sampling, tau_points.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Matsubara Sampling
    spir_sampling *matsubara_sampling = spir_matsubara_sampling_new(basis);
    REQUIRE(matsubara_sampling != nullptr);
    int32_t num_matsubara_points;
    status = spir_sampling_get_num_points(matsubara_sampling, &num_matsubara_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    Eigen::Vector<int32_t, Eigen::Dynamic> matsubara_points(num_matsubara_points);
    status = spir_sampling_get_matsubara_points(matsubara_sampling, matsubara_points.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // DLR
    spir_dlr *dlr = spir_dlr_new(basis);
    REQUIRE(dlr != nullptr);
    int32_t npoles;
    status = spir_dlr_get_num_poles(dlr, &npoles);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(npoles >= 0);
    REQUIRE(npoles >= basis_size);
    Eigen::VectorXd poles(npoles);
    status = spir_dlr_get_poles(dlr, poles.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Calculate total size of extra dimensions
    Eigen::Index extra_size = std::accumulate(
        extra_dims.begin(), extra_dims.end(), 1, std::multiplies<>());

    // Generate random DLR coefficients
    Eigen::Tensor<double, ndim, ORDER> coeffs_targetdim0(
        _get_dims<ndim>(npoles, extra_dims, target_dim));
    std::mt19937 gen(982743);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    {
        Eigen::TensorMap<Eigen::Tensor<double, 2, ORDER>> coeffs_2d(
            coeffs_targetdim0.data(), npoles, extra_size);
        for (Eigen::Index i = 0; i < npoles; ++i) {
            for (Eigen::Index j = 0; j < extra_size; ++j) {
                coeffs_2d(i, j) =
                    (2.0 * dis(gen) - 1.0) * std::sqrt(std::abs(poles(i)));
            }
        }
    }
    REQUIRE(poles.array().abs().maxCoeff() <= wmax);

    // Move the axis for the poles from the first to the target dimension
    Eigen::Tensor<double, ndim, ORDER> coeffs =
        sparseir::movedim(coeffs_targetdim0, 0, target_dim);

    // Convert DLR coefficients to IR coefficients
    // TODO: Extend to Tensor
    Eigen::Tensor<double, ndim, ORDER> g_IR(
        _get_dims<ndim>(basis_size, extra_dims, target_dim));
    status = spir_dlr_to_IR(dlr, order, ndim, &basis_size, target_dim,
                            coeffs.data(), g_IR.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // DLR basis functions
    spir_funcs *dlr_u;
    status = spir_dlr_get_u(dlr, &dlr_u);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    spir_matsubara_funcs *dlr_uhat;
    status = spir_dlr_get_uhat(dlr, &dlr_uhat);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // IR basis functions
    spir_funcs *ir_u;
    status = spir_finite_temp_basis_get_u(basis, &ir_u);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    spir_matsubara_funcs *ir_uhat;
    status = spir_finite_temp_basis_get_uhat(basis, &ir_uhat);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Compare the Greens function at all tau points between IR and DLR
    Eigen::Tensor<double, ndim, ORDER> gtau_from_IR =
        _evaluate_gtau<double, ndim, ORDER>(
            g_IR, ir_u, target_dim, tau_points);
    Eigen::Tensor<double, ndim, ORDER> gtau_from_DLR =
        _evaluate_gtau<double, ndim, ORDER>(
            coeffs, dlr_u, target_dim, tau_points);
    REQUIRE(compare_tensors_with_relative_error<double, ndim, ORDER>(
        gtau_from_IR, gtau_from_DLR, epsilon));

    // TODO:
    // - Compare the Greens function at Matsubara frequencies
    // - Check the accuracy of Fourier transform

    spir_destroy_finite_temp_basis(basis);
    spir_destroy_dlr(dlr);
    spir_destroy_funcs(dlr_u);
    spir_destroy_funcs(ir_u);
    spir_destroy_sampling(tau_sampling);
}

TEST_CASE("Integration Test", "[cinterface]")
{
    std::vector<int> extra_dims = {};
    integration_test<sparseir::Fermionic, sparseir::LogisticKernel, 1,
                     Eigen::ColMajor>(1.0, 10.0, 1e-10, extra_dims, 0,
                                      SPIR_ORDER_COLUMN_MAJOR);
}