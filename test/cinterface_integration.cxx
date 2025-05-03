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

template <int ndim, typename IntType = Eigen::Index>
std::array<IntType, ndim> _get_dims(int target_dim_size,
                                    const std::vector<int> &extra_dims,
                                    int target_dim)
{
    std::array<IntType, ndim> dims;
    dims[target_dim] = static_cast<IntType>(target_dim_size);
    int pos = 0;
    for (int i = 0; i < extra_dims.size(); ++i) {
        if (i == target_dim) {
            continue;
        }
        dims[i] = static_cast<IntType>(extra_dims[pos]);
        ++pos;
    }
    return dims;
}

// Helper function to evaluate basis functions at multiple points
template <typename T, Eigen::StorageOptions ORDER>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER>
_evaluate_basis_functions(const spir_funcs *u, const Eigen::VectorXd &x_values)
{
    int32_t status;
    int32_t funcs_size;
    status = spir_funcs_get_size(u, &funcs_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER> u_eval_mat(
        x_values.size(), funcs_size);
    for (Eigen::Index i = 0; i < x_values.size(); ++i) {
        Eigen::VectorXd u_eval(funcs_size);
        status = spir_evaluate_funcs(u, x_values(i), u_eval.data());
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        u_eval_mat.row(i) = u_eval.transpose().cast<T>();
    }
    return u_eval_mat;
}

// Helper function to evaluate Matsubara basis functions at multiple frequencies
template <typename T, Eigen::StorageOptions ORDER>
Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, ORDER>
_evaluate_matsubara_basis_functions(const spir_matsubara_funcs *uhat,
                                    const Eigen::VectorXi &matsubara_indices)
{
    int32_t status;
    int32_t funcs_size;
    status = spir_matsubara_funcs_get_size(uhat, &funcs_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Allocate output matrix with shape (nfreqs, nfuncs)
    Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, ORDER>
        uhat_eval_mat(matsubara_indices.size(), funcs_size);

    // Create a non-const copy of the Matsubara indices
    std::vector<int32_t> freq_indices(matsubara_indices.data(),
                                      matsubara_indices.data() +
                                          matsubara_indices.size());

    // Evaluate all frequencies at once
    status = spir_evaluate_matsubara_funcs(
        uhat,
        ORDER == Eigen::ColMajor ? SPIR_ORDER_COLUMN_MAJOR
                                 : SPIR_ORDER_ROW_MAJOR,
        matsubara_indices.size(), freq_indices.data(),
        reinterpret_cast<c_complex *>(uhat_eval_mat.data()));
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    return uhat_eval_mat;
}

// Helper function to perform the tensor transformation
template <typename T, int ndim, Eigen::StorageOptions ORDER,
          typename BasisEvalType>
Eigen::Tensor<typename BasisEvalType::Scalar, ndim, ORDER>
_transform_coefficients(const Eigen::Tensor<T, ndim, ORDER> &coeffs,
                        const BasisEvalType &basis_eval, int target_dim)
{
    // Move target dimension to the first position
    Eigen::Tensor<T, ndim, ORDER> coeffs_targetdim0 =
        sparseir::movedim(coeffs, target_dim, 0);

    // Calculate the size of extra dimensions
    Eigen::Index extra_size = 1;
    for (int i = 1; i < ndim; ++i) {
        extra_size *= coeffs_targetdim0.dimension(i);
    }

    // Create result tensor with correct dimensions
    std::array<Eigen::Index, ndim> dims;
    dims[0] = basis_eval.rows();
    for (int i = 1; i < ndim; ++i) {
        dims[i] = coeffs_targetdim0.dimension(i);
    }
    Eigen::Tensor<typename BasisEvalType::Scalar, ndim, ORDER> result(dims);

    // Map tensors to matrices for multiplication
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER>>
        coeffs_mat(coeffs_targetdim0.data(), coeffs_targetdim0.dimension(0),
                   extra_size);
    Eigen::Map<Eigen::Matrix<typename BasisEvalType::Scalar, Eigen::Dynamic,
                             Eigen::Dynamic, ORDER>>
        result_mat(result.data(), basis_eval.rows(), extra_size);

    // Perform matrix multiplication with consistent type
    result_mat =
        basis_eval * coeffs_mat.template cast<typename BasisEvalType::Scalar>();

    // Move dimensions back to original order
    return sparseir::movedim(result, 0, target_dim);
}

template <typename T, int ndim, Eigen::StorageOptions ORDER>
Eigen::Tensor<T, ndim, ORDER>
_evaluate_gtau(const Eigen::Tensor<T, ndim, ORDER> &coeffs, const spir_funcs *u,
               int target_dim, const Eigen::VectorXd &x_values)
{
    auto u_eval_mat = _evaluate_basis_functions<T, ORDER>(u, x_values);
    return _transform_coefficients<T, ndim, ORDER>(coeffs, u_eval_mat,
                                                   target_dim);
}

template <typename T, int ndim, Eigen::StorageOptions ORDER>
Eigen::Tensor<std::complex<T>, ndim, ORDER>
_evaluate_giw(const Eigen::Tensor<T, ndim, ORDER> &coeffs,
              const spir_matsubara_funcs *uhat, int target_dim,
              const Eigen::VectorXi &matsubara_indices)
{
    auto uhat_eval_mat =
        _evaluate_matsubara_basis_functions<T, ORDER>(uhat, matsubara_indices);

    auto result = _transform_coefficients<T, ndim, ORDER>(coeffs, uhat_eval_mat,
                                                          target_dim);
    return result;
}

template <typename T, int ndim, Eigen::StorageOptions ORDER>
bool compare_tensors_with_relative_error(const Eigen::Tensor<T, ndim, ORDER> &a,
                                         const Eigen::Tensor<T, ndim, ORDER> &b,
                                         double tol)
{
    // Convert to double tensor for absolute values
    Eigen::Tensor<double, ndim, ORDER> diff = (a - b).abs();
    Eigen::Tensor<double, ndim, ORDER> ref = a.abs();

    // Map tensors to matrices and use maxCoeff
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> diff_vec(
        diff.data(), diff.size());
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> ref_vec(
        ref.data(), ref.size());

    double max_diff = diff_vec.maxCoeff();
    double max_ref = ref_vec.maxCoeff();

    // debug
    if (max_diff > tol * max_ref) {
        std::cout << "max_diff: " << max_diff << std::endl;
        std::cout << "max_ref: " << max_ref << std::endl;
        std::cout << "tol " << tol << std::endl;
    }

    return max_diff <= tol * max_ref;
}

template <typename K>
spir_kernel* _kernel_new(double lambda);

template <>
spir_kernel* _kernel_new<sparseir::LogisticKernel>(double lambda)
{  
    return spir_logistic_kernel_new(lambda);
}

template <>
spir_kernel* _kernel_new<sparseir::RegularizedBoseKernel>(double lambda)
{
    return spir_regularized_bose_kernel_new(lambda);
}

/*
T: double or std::complex<double>, scalar type of coeffs
TODO: we need to test positive only mode. A different function is needed?
*/
template <typename S, typename K, int ndim, Eigen::StorageOptions ORDER>
void integration_test(double beta, double wmax, double epsilon,
                      const std::vector<int> &extra_dims, int target_dim,
                      const spir_order_type order, double tol)
{
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
    spir_kernel* kernel = _kernel_new<K>(beta * wmax);
    spir_sve_result* sve = spir_sve_result_new(kernel, epsilon);
    spir_finite_temp_basis* basis = spir_finite_temp_basis_new_with_sve(stat, beta, wmax, kernel, sve);

    REQUIRE(basis != nullptr);
    int32_t basis_size;
    status = spir_finite_temp_basis_get_size(basis, &basis_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Tau Sampling
    std::cout << "Tau sampling" << std::endl;
    spir_sampling *tau_sampling = spir_tau_sampling_new(basis);
    REQUIRE(tau_sampling != nullptr);
    int32_t num_tau_points;
    status = spir_sampling_get_num_points(tau_sampling, &num_tau_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    Eigen::VectorXd tau_points(num_tau_points);
    status = spir_sampling_get_tau_points(tau_sampling, tau_points.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(num_tau_points >= basis_size);

    // Matsubara Sampling
    std::cout << "Matsubara sampling" << std::endl;
    spir_sampling *matsubara_sampling = spir_matsubara_sampling_new(basis);
    REQUIRE(matsubara_sampling != nullptr);
    int32_t num_matsubara_points;
    status =
        spir_sampling_get_num_points(matsubara_sampling, &num_matsubara_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    Eigen::Vector<int32_t, Eigen::Dynamic> matsubara_points(
        num_matsubara_points);
    status = spir_sampling_get_matsubara_points(matsubara_sampling,
                                                matsubara_points.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(num_matsubara_points >= basis_size);

    // DLR
    std::cout << "DLR" << std::endl;
    spir_dlr *dlr = spir_dlr_new(basis);
    REQUIRE(dlr != nullptr);
    int32_t npoles;
    status = spir_dlr_get_num_poles(dlr, &npoles);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
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
        coeffs_2d(0, 0) = 1.0;
        for (Eigen::Index i = 1; i < npoles; ++i) {
            coeffs_2d(i, 0) = 0.0;
        }

    }
    REQUIRE(poles.array().abs().maxCoeff() <= wmax);
    //std::cout << "poles: " << poles << std::endl;

    // Move the axis for the poles from the first to the target dimension
    Eigen::Tensor<double, ndim, ORDER> coeffs =
        sparseir::movedim(coeffs_targetdim0, 0, target_dim);

    // Convert DLR coefficients to IR coefficients
    // TODO: Extend to Tensor
    Eigen::Tensor<double, ndim, ORDER> g_IR(
        _get_dims<ndim>(basis_size, extra_dims, target_dim));
    status = spir_dlr_to_IR(dlr, order, ndim,
                            _get_dims<ndim, int32_t>(npoles, extra_dims, target_dim).data(), target_dim,
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
    std::cout << "Evaluate Greens function at all tau points between IR and DLR" << std::endl;
    Eigen::Tensor<double, ndim, ORDER> gtau_from_IR =
        _evaluate_gtau<double, ndim, ORDER>(g_IR, ir_u, target_dim, tau_points);
    Eigen::Tensor<double, ndim, ORDER> gtau_from_DLR =
        _evaluate_gtau<double, ndim, ORDER>(coeffs, dlr_u, target_dim,
                                            tau_points);
    Eigen::Tensor<double, ndim, ORDER> gtau_diff = (gtau_from_IR - gtau_from_DLR).abs();
    std::cout << "gtau_from_IR: " << gtau_from_IR << std::endl;
    std::cout << "gtau_from_DLR: " << gtau_from_DLR << std::endl;
    std::cout << "gtau_diff: " << gtau_diff << std::endl;
    REQUIRE(compare_tensors_with_relative_error<double, ndim, ORDER>(
        gtau_from_IR, gtau_from_DLR, tol));

    // Compare the Greens function at all Matsubara frequencies between IR and
    // DLR
    Eigen::Tensor<std::complex<double>, ndim, ORDER> giw_from_IR =
        _evaluate_giw<double, ndim, ORDER>(g_IR, ir_uhat, target_dim,
                                           matsubara_points);
    Eigen::Tensor<std::complex<double>, ndim, ORDER> giw_from_DLR =
        _evaluate_giw<double, ndim, ORDER>(coeffs, dlr_uhat, target_dim,
                                           matsubara_points);
    REQUIRE(
        compare_tensors_with_relative_error<std::complex<double>, ndim, ORDER>(
            giw_from_IR, giw_from_DLR, tol));

    auto dims_matsubara =
        _get_dims<ndim, int32_t>(num_matsubara_points, extra_dims, target_dim);
    auto dims_IR = _get_dims<ndim, int32_t>(basis_size, extra_dims, target_dim);
    auto dims_tau =
        _get_dims<ndim, int32_t>(num_tau_points, extra_dims, target_dim);

    Eigen::Tensor<std::complex<double>, ndim, ORDER> gIR(
        _get_dims<ndim, Eigen::Index>(basis_size, extra_dims, target_dim));
    Eigen::Tensor<std::complex<double>, ndim, ORDER> gIR2(
        _get_dims<ndim, Eigen::Index>(basis_size, extra_dims, target_dim));
    Eigen::Tensor<std::complex<double>, ndim, ORDER> gtau(
        _get_dims<ndim, Eigen::Index>(num_tau_points, extra_dims, target_dim));
    Eigen::Tensor<std::complex<double>, ndim, ORDER> giw_reconst(
        _get_dims<ndim, Eigen::Index>(num_matsubara_points, extra_dims,
                                      target_dim));

    // Matsubara -> IR
    status = spir_sampling_fit_zz(
        matsubara_sampling, order, ndim, dims_matsubara.data(), target_dim,
        reinterpret_cast<const c_complex *>(giw_from_DLR.data()),
        reinterpret_cast<c_complex *>(gIR.data()));
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // IR -> tau
    status = spir_sampling_evaluate_zz(
        tau_sampling, order, ndim, dims_IR.data(), target_dim,
        reinterpret_cast<const c_complex *>(gIR.data()),
        reinterpret_cast<c_complex *>(gtau.data()));
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    //REQUIRE(
        //compare_tensors_with_relative_error<std::complex<double>, ndim, ORDER>(
            //gtau, gtau_from_DLR, tol));

    // tau -> IR
    status = spir_sampling_fit_zz(
        tau_sampling, order, ndim, dims_tau.data(), target_dim,
        reinterpret_cast<const c_complex *>(gtau.data()),
        reinterpret_cast<c_complex *>(gIR2.data()));
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // IR -> Matsubara
    status = spir_sampling_evaluate_zz(
        matsubara_sampling, order, ndim, dims_IR.data(), target_dim,
        reinterpret_cast<const c_complex *>(gIR2.data()),
        reinterpret_cast<c_complex *>(giw_reconst.data()));
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    REQUIRE(
        compare_tensors_with_relative_error<std::complex<double>, ndim, ORDER>(
            giw_from_DLR, giw_reconst, tol));

    spir_destroy_finite_temp_basis(basis);
    spir_destroy_dlr(dlr);
    spir_destroy_funcs(dlr_u);
    spir_destroy_funcs(ir_u);
    spir_destroy_sampling(tau_sampling);
}

TEST_CASE("Integration Test", "[cinterface]")
{
    std::vector<int> extra_dims = {};
    double beta = 1e+4;
    double wmax = 2.0;
    double epsilon = 1e-10;

    double tol = 10 * epsilon;

    //std::cout << "Integration test for fermionic LogisticKernel" << std::endl;
    //integration_test<sparseir::Fermionic, sparseir::LogisticKernel, 1,
                    //Eigen::ColMajor>(beta, wmax, epsilon, extra_dims, 0,
                                      //SPIR_ORDER_COLUMN_MAJOR, tol);

    std::cout << "Integration test for bosonic LogisticKernel" << std::endl;
    integration_test<sparseir::Bosonic, sparseir::LogisticKernel, 1,
                    Eigen::ColMajor>(beta, wmax, epsilon, extra_dims, 0,
                                      SPIR_ORDER_COLUMN_MAJOR, tol);
//
    //std::cout << "Integration test for bosonic RegularizedBoseKernel" << std::endl;
    //integration_test<sparseir::Bosonic, sparseir::RegularizedBoseKernel, 1,
                    //Eigen::ColMajor>(beta, wmax, epsilon, extra_dims, 0,
                                      //SPIR_ORDER_COLUMN_MAJOR, tol);
}