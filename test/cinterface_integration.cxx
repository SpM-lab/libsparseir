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
    for (int i = 0; i < ndim; ++i) {
        if (i == target_dim) {
            continue;
        }
        dims[i] = static_cast<IntType>(extra_dims[pos]);
        ++pos;
    }
    return dims;
}

TEST_CASE("Test _get_dims", "[cinterface]")
{
    std::vector<int> extra_dims = {2,3,4};
    {
        int32_t target_dim = 0;
        auto dims = _get_dims<4>(100, extra_dims, target_dim);
        REQUIRE(dims[0] == 100);
        REQUIRE(dims[1] == 2);
        REQUIRE(dims[2] == 3);
        REQUIRE(dims[3] == 4);
    }
    {
        int32_t target_dim = 1;
        auto dims = _get_dims<4>(100, extra_dims, target_dim);
        REQUIRE(dims[0] == 2);
        REQUIRE(dims[1] == 100);
        REQUIRE(dims[2] == 3);
        REQUIRE(dims[3] == 4);
    }
    {
        int32_t target_dim = 2;
        auto dims = _get_dims<4>(100, extra_dims, target_dim);
        REQUIRE(dims[0] == 2);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 100);
        REQUIRE(dims[3] == 4);
    }
    {
        int32_t target_dim = 3;
        auto dims = _get_dims<4>(100, extra_dims, target_dim);
        REQUIRE(dims[0] == 2);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 4);
        REQUIRE(dims[3] == 100);
    }
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
template <Eigen::StorageOptions ORDER>
Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, ORDER>
_evaluate_matsubara_basis_functions(const spir_matsubara_funcs *uhat,
                                    const Eigen::VectorXi &matsubara_indices)
{
    int32_t status;
    int32_t funcs_size;
    status = spir_matsubara_funcs_get_size(uhat, &funcs_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Allocate output matrix with shape (nfreqs, nfuncs)
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, ORDER>
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
Eigen::Tensor<decltype(T() * typename BasisEvalType::Scalar()), ndim, ORDER>
_transform_coefficients(const Eigen::Tensor<T, ndim, ORDER> &coeffs,
                        const BasisEvalType &basis_eval, int target_dim)
{
    using PromotedType = decltype(T() * typename BasisEvalType::Scalar());

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
    Eigen::Tensor<PromotedType, ndim, ORDER> result(dims);

    // Map tensors to matrices for multiplication
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER>>
        coeffs_mat(coeffs_targetdim0.data(), coeffs_targetdim0.dimension(0),
                   extra_size);
    Eigen::Map<Eigen::Matrix<PromotedType, Eigen::Dynamic,
                             Eigen::Dynamic, ORDER>>
        result_mat(result.data(), basis_eval.rows(), extra_size);

    // Perform matrix multiplication with proper type promotion
    result_mat = basis_eval * coeffs_mat.template cast<PromotedType>();

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
Eigen::Tensor<std::complex<double>, ndim, ORDER>
_evaluate_giw(const Eigen::Tensor<T, ndim, ORDER> &coeffs,
              const spir_matsubara_funcs *uhat, int target_dim,
              const Eigen::VectorXi &matsubara_indices)
{
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, ORDER> uhat_eval_mat =
        _evaluate_matsubara_basis_functions<ORDER>(uhat, matsubara_indices);

    Eigen::Tensor<std::complex<double>, ndim, ORDER> result =
        _transform_coefficients<T, ndim, ORDER>(coeffs, uhat_eval_mat,
                                                          target_dim);
    return result;
}

template <typename T, int ndim, Eigen::StorageOptions ORDER>
bool compare_tensors_with_relative_error(const Eigen::Tensor<T, ndim, ORDER> &a,
                                         const Eigen::Tensor<T, ndim, ORDER> &b,
                                         double tol)
{
    // Convert to double tensor for absolute values
    Eigen::Tensor<double, ndim, ORDER> diff(a.dimensions());
    Eigen::Tensor<double, ndim, ORDER> ref(a.dimensions());

    // Compute absolute values element-wise
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        diff.data()[i] = std::abs(a.data()[i] - b.data()[i]);
        ref.data()[i] = std::abs(a.data()[i]);
    }

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
    int32_t status;
    spir_kernel* kernel = spir_logistic_kernel_new(lambda, &status);
    return kernel;
}

template <>
spir_kernel* _kernel_new<sparseir::RegularizedBoseKernel>(double lambda)
{
    int32_t status;
    spir_kernel* kernel = spir_regularized_bose_kernel_new(lambda, &status);
    return kernel;
}

// Add these template functions before the integration_test function
template <typename T>
T generate_random_coeff(double random_value, double pole) {
    return (2.0 * random_value - 1.0) * std::sqrt(std::abs(pole));
}

template <>
std::complex<double> generate_random_coeff<std::complex<double>>(double random_value, double pole) {
    return std::complex<double>(
        (2.0 * random_value - 1.0) * std::sqrt(std::abs(pole)),
        (2.0 * random_value - 1.0) * std::sqrt(std::abs(pole))
    );
}

// Add these template functions before the integration_test function
template <typename T>
int32_t dlr_to_IR(spir_dlr* dlr, spir_order_type order, int32_t ndim,
                  const int32_t* dims, int32_t target_dim,
                  const T* coeffs, T* g_IR) {
    if (std::is_same<T, double>::value) {
        return spir_dlr_to_IR_dd(dlr, order, ndim, dims, target_dim,
                                reinterpret_cast<const double*>(coeffs),
                                reinterpret_cast<double*>(g_IR));
    } else if (std::is_same<T, std::complex<double>>::value) {
        return spir_dlr_to_IR_zz(dlr, order, ndim, dims, target_dim,
                                reinterpret_cast<const c_complex*>(coeffs),
                                reinterpret_cast<c_complex*>(g_IR));
    }
    return SPIR_INVALID_ARGUMENT;
}


template <typename T>
int32_t dlr_from_IR(spir_dlr* dlr, spir_order_type order, int32_t ndim,
                  const int32_t* dims, int32_t target_dim,
                  const T* coeffs, T* g_IR) {
    if (std::is_same<T, double>::value) {
        return spir_dlr_from_IR_dd(dlr, order, ndim, dims, target_dim,
                                reinterpret_cast<const double*>(coeffs),
                                reinterpret_cast<double*>(g_IR));
    } else if (std::is_same<T, std::complex<double>>::value) {
        return spir_dlr_from_IR_zz(dlr, order, ndim, dims, target_dim,
                                reinterpret_cast<const c_complex*>(coeffs),
                                reinterpret_cast<c_complex*>(g_IR));
    }
    return SPIR_INVALID_ARGUMENT;
}

template <typename T>
int32_t _tau_sampling_evaluate(spir_sampling* sampling, spir_order_type order, int32_t ndim,
                         const int32_t* dims, int32_t target_dim,
                         const T* gIR, T* gtau) {
    if (std::is_same<T, double>::value) {
        return spir_sampling_evaluate_dd(sampling, order, ndim, dims, target_dim,
                                       reinterpret_cast<const double*>(gIR),
                                       reinterpret_cast<double*>(gtau));
    } else if (std::is_same<T, std::complex<double>>::value) {
        return spir_sampling_evaluate_zz(sampling, order, ndim, dims, target_dim,
                                       reinterpret_cast<const c_complex*>(gIR),
                                       reinterpret_cast<c_complex*>(gtau));
    }
    return SPIR_INVALID_ARGUMENT;
}

template <typename T>
int32_t _tau_sampling_fit(spir_sampling* sampling, spir_order_type order, int32_t ndim,
                         const int32_t* dims, int32_t target_dim,
                         const T* gtau, T* gIR) {
    if (std::is_same<T, double>::value) {
        return spir_sampling_fit_dd(sampling, order, ndim, dims, target_dim,
                                  reinterpret_cast<const double*>(gtau),
                                  reinterpret_cast<double*>(gIR));
    } else if (std::is_same<T, std::complex<double>>::value) {
        return spir_sampling_fit_zz(sampling, order, ndim, dims, target_dim,
                                  reinterpret_cast<const c_complex*>(gtau),
                                  reinterpret_cast<c_complex*>(gIR));
    }
    return SPIR_INVALID_ARGUMENT;
}

template <typename T>
int32_t _matsubara_sampling_evaluate(spir_sampling* sampling, spir_order_type order, int32_t ndim,
                                   const int32_t* dims, int32_t target_dim,
                                   const T* gIR, std::complex<double>* giw) {
    if (std::is_same<T, double>::value) {
        return spir_sampling_evaluate_dz(sampling, order, ndim, dims, target_dim,
                                       reinterpret_cast<const double*>(gIR),
                                       reinterpret_cast<c_complex*>(giw));
    } else if (std::is_same<T, std::complex<double>>::value) {
        return spir_sampling_evaluate_zz(sampling, order, ndim, dims, target_dim,
                                       reinterpret_cast<const c_complex*>(gIR),
                                       reinterpret_cast<c_complex*>(giw));
    }
    return SPIR_INVALID_ARGUMENT;
}

template <typename T, int ndim, Eigen::StorageOptions ORDER>
struct tensor_converter {
    static void convert(const Eigen::Tensor<std::complex<double>, ndim, ORDER>& src, Eigen::Tensor<T, ndim, ORDER>& dst) {
        dst = src.template cast<T>();
    }
};

template <int ndim, Eigen::StorageOptions ORDER>
struct tensor_converter<double, ndim, ORDER> {
    static void convert(const Eigen::Tensor<std::complex<double>, ndim, ORDER>& src, Eigen::Tensor<double, ndim, ORDER>& dst) {
        dst = src.real();
    }
};

/*
T: double or std::complex<double>, scalar type of coeffs
*/
template <typename T, typename S, typename K, int ndim, Eigen::StorageOptions ORDER>
void integration_test(double beta, double wmax, double epsilon,
                      const std::vector<int> &extra_dims, int target_dim,
                      const spir_order_type order, double tol, bool positive_only)
{
    // positive_only is not supported for complex numbers
    REQUIRE (!(std::is_same<T, std::complex<double>>::value && positive_only));

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
    spir_sve_result* sve = spir_sve_result_new(kernel, epsilon, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(sve != nullptr);

    spir_basis *basis = spir_finite_temp_basis_new(stat, beta, wmax, epsilon, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    REQUIRE(basis != nullptr);

    int32_t basis_size;
    status = spir_finite_temp_basis_get_size(basis, &basis_size);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Tau Sampling
    std::cout << "Tau sampling" << std::endl;
    spir_sampling *tau_sampling = spir_tau_sampling_new(basis, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
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
    spir_sampling *matsubara_sampling = spir_matsubara_sampling_new(basis, positive_only, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
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
    if (positive_only) {
        REQUIRE(num_matsubara_points >= basis_size / 2);
    } else {
        REQUIRE(num_matsubara_points >= basis_size);
    }

    // DLR
    std::cout << "DLR" << std::endl;
    spir_dlr *dlr = spir_dlr_new(basis, &status);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
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
    Eigen::Tensor<T, ndim, ORDER> coeffs_targetdim0(
        _get_dims<ndim>(npoles, extra_dims, 0));
    std::mt19937 gen(982743);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    {
        Eigen::TensorMap<Eigen::Tensor<T, 2, ORDER>> coeffs_2d(
            coeffs_targetdim0.data(), npoles, extra_size);
        for (Eigen::Index i = 0; i < npoles; ++i) {
            for (Eigen::Index j = 0; j < extra_size; ++j) {
                coeffs_2d(i, j) = generate_random_coeff<T>(dis(gen), poles(i));
            }
        }
    }
    REQUIRE(poles.array().abs().maxCoeff() <= wmax);

    // Move the axis for the poles from the first to the target dimension
    Eigen::Tensor<T, ndim, ORDER> coeffs =
        sparseir::movedim(coeffs_targetdim0, 0, target_dim);
    // Convert DLR coefficients to IR coefficients
    Eigen::Tensor<T, ndim, ORDER> g_IR(
        _get_dims<ndim>(basis_size, extra_dims, target_dim));
    status = dlr_to_IR(dlr, order, ndim,
                      _get_dims<ndim, int32_t>(npoles, extra_dims, target_dim).data(), target_dim,
                      coeffs.data(), g_IR.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // Convert IR coefficients to DLR coefficients
    Eigen::Tensor<T, ndim, ORDER> g_DLR_reconst(
        _get_dims<ndim>(basis_size, extra_dims, target_dim));
    status = dlr_from_IR(dlr, order, ndim,
                        _get_dims<ndim, int32_t>(npoles, extra_dims, target_dim).data(), target_dim,
                        g_IR.data(), g_DLR_reconst.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // From_IR C API
    Eigen::Tensor<T, ndim, ORDER> g_dlr(
        _get_dims<ndim>(basis_size, extra_dims, target_dim));
    status = dlr_from_IR(dlr, order, ndim,
                            _get_dims<ndim, int32_t>(basis_size, extra_dims, target_dim).data(), target_dim,
                            g_IR.data(), g_dlr.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    std::cout << "g_dlr: " << g_dlr << std::endl;

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
    Eigen::Tensor<T, ndim, ORDER> gtau_from_IR =
        _evaluate_gtau<T, ndim, ORDER>(g_IR, ir_u, target_dim, tau_points);
    Eigen::Tensor<T, ndim, ORDER> gtau_from_DLR =
        _evaluate_gtau<T, ndim, ORDER>(coeffs, dlr_u, target_dim,
                                            tau_points);
    Eigen::Tensor<T, ndim, ORDER> gtau_from_DLR_reconst =
        _evaluate_gtau<T, ndim, ORDER>(g_DLR_reconst, dlr_u, target_dim,
                                            tau_points);
    REQUIRE(compare_tensors_with_relative_error<T, ndim, ORDER>(
        gtau_from_IR, gtau_from_DLR, tol));
    REQUIRE(compare_tensors_with_relative_error<T, ndim, ORDER>(
        gtau_from_IR, gtau_from_DLR_reconst, tol));

    // Compare the Greens function at all Matsubara frequencies between IR and
    // DLR
    Eigen::Tensor<std::complex<double>, ndim, ORDER> giw_from_IR =
        _evaluate_giw<T, ndim, ORDER>(g_IR, ir_uhat, target_dim,
                                           matsubara_points);
    Eigen::Tensor<std::complex<double>, ndim, ORDER> giw_from_DLR =
        _evaluate_giw<T, ndim, ORDER>(coeffs, dlr_uhat, target_dim,
                                           matsubara_points);
    REQUIRE(
        compare_tensors_with_relative_error<std::complex<double>, ndim, ORDER>(
            giw_from_IR, giw_from_DLR, tol));

    auto dims_matsubara =
        _get_dims<ndim, int32_t>(num_matsubara_points, extra_dims, target_dim);
    auto dims_IR = _get_dims<ndim, int32_t>(basis_size, extra_dims, target_dim);
    auto dims_tau =
        _get_dims<ndim, int32_t>(num_tau_points, extra_dims, target_dim);

    Eigen::Tensor<T, ndim, ORDER> gIR(
        _get_dims<ndim, Eigen::Index>(basis_size, extra_dims, target_dim));
    Eigen::Tensor<T, ndim, ORDER> gIR2(
        _get_dims<ndim, Eigen::Index>(basis_size, extra_dims, target_dim));
    Eigen::Tensor<T, ndim, ORDER> gtau(
        _get_dims<ndim, Eigen::Index>(num_tau_points, extra_dims, target_dim));
    Eigen::Tensor<std::complex<double>, ndim, ORDER> giw_reconst(
        _get_dims<ndim, Eigen::Index>(num_matsubara_points, extra_dims,
                                      target_dim));

    // Matsubara -> IR
    {
        Eigen::Tensor<std::complex<double>, ndim, ORDER>
            gIR_work( _get_dims<ndim, Eigen::Index>(basis_size, extra_dims, target_dim));
        status = spir_sampling_fit_zz(
            matsubara_sampling, order, ndim, dims_matsubara.data(), target_dim,
            reinterpret_cast<const c_complex *>(giw_from_DLR.data()),
            reinterpret_cast<c_complex *>(gIR_work.data()));
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        tensor_converter<T, ndim, ORDER>::convert(gIR_work, gIR);
    }

    // IR -> tau
    status = _tau_sampling_evaluate<T>(
        tau_sampling, order, ndim, dims_IR.data(), target_dim,
        gIR.data(), gtau.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // tau -> IR
    status = _tau_sampling_fit<T>(
        tau_sampling, order, ndim, dims_tau.data(), target_dim,
        gtau.data(), gIR2.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    // IR -> Matsubara
    status = _matsubara_sampling_evaluate<T>(
        matsubara_sampling, order, ndim, dims_IR.data(), target_dim,
        gIR2.data(), giw_reconst.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);


    Eigen::Tensor<std::complex<double>, ndim, ORDER> giv_ref(
        _get_dims<ndim, Eigen::Index>(num_matsubara_points, extra_dims,
                                      target_dim));

    status = _matsubara_sampling_evaluate<T>(
        matsubara_sampling, order, ndim, _get_dims<ndim, int32_t>(basis_size, extra_dims, target_dim).data(), target_dim,
        g_IR.data(),
        giv_ref.data()
    );
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    std::cout << "giv_ref: " << giv_ref << std::endl;

    //Eigen::Tensor<std::complex<double>, ndim, ORDER> giv(
        //_get_dims<ndim, Eigen::Index>(num_matsubara_points, extra_dims,
                                      //target_dim));
//
    //spir_sampling *smpl_for_dlr = spir_matsubara_sampling_new(dlr, num_matsubara_points, matsubara_points.data(), positive_only, &status);
    //REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    //REQUIRE(smpl_for_dlr != nullptr);
//
    //status = _matsubara_sampling_evaluate<T>(
        //smpl_for_dlr, order, ndim, _get_dims<ndim, int32_t>(basis_size, extra_dims, target_dim).data(), target_dim,
        //g_dlr.data(),
        //giv.data()
    //);
    //REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    //REQUIRE(compare_tensors_with_relative_error<std::complex<double>, ndim, ORDER>(giv, giv_ref, 300 * epsilon));
//
    //REQUIRE(
        //compare_tensors_with_relative_error<std::complex<double>, ndim, ORDER>(
            //giw_from_DLR, giw_reconst, tol));

    spir_basis_destroy(basis);
    spir_dlr_destroy(dlr);
    spir_funcs_destroy(dlr_u);
    spir_funcs_destroy(ir_u);
    spir_sampling_destroy(tau_sampling);
}

TEST_CASE("Integration Test", "[cinterface]") {
    double beta = 1e+4;
    beta = 10.0;
    double wmax = 2.0;
    double epsilon = 1e-10;

    double tol = 10 * epsilon;


    for (bool positive_only : {false, true}) {
        std::cout << "positive_only = " << positive_only << std::endl;
        {
            std::vector<int> extra_dims = {};
            std::cout << "Integration test for bosonic LogisticKernel" << std::endl;
            integration_test<double, sparseir::Bosonic, sparseir::LogisticKernel, 1,
                           Eigen::ColMajor>(beta, wmax, epsilon, extra_dims, 0,
                                          SPIR_ORDER_COLUMN_MAJOR, tol, positive_only);

            if (!positive_only) {
                integration_test<std::complex<double>, sparseir::Bosonic, sparseir::LogisticKernel, 1,
                               Eigen::ColMajor>(beta, wmax, epsilon, extra_dims, 0,
                                              SPIR_ORDER_COLUMN_MAJOR, tol, positive_only);
            }
        }

        {
            int32_t target_dim = 0;
            std::vector<int> extra_dims = {};
            std::cout << "Integration test for bosonic LogisticKernel, ColMajor, target_dim = " << target_dim << std::endl;
            integration_test<double, sparseir::Bosonic, sparseir::LogisticKernel, 1,
                           Eigen::ColMajor>(beta, wmax, epsilon, extra_dims, target_dim,
                                          SPIR_ORDER_COLUMN_MAJOR, tol, positive_only);
            if (!positive_only) {
                integration_test<std::complex<double>, sparseir::Bosonic, sparseir::LogisticKernel, 1,
                               Eigen::ColMajor>(beta, wmax, epsilon, extra_dims, target_dim,
                                              SPIR_ORDER_COLUMN_MAJOR, tol, positive_only);
            }
        }

        {
            int32_t target_dim = 0;
            std::vector<int> extra_dims = {};
            std::cout << "Integration test for bosonic LogisticKernel, RowMajor, target_dim = " << target_dim << std::endl;
            integration_test<double, sparseir::Bosonic, sparseir::LogisticKernel, 1,
                           Eigen::RowMajor>(beta, wmax, epsilon, extra_dims, target_dim,
                                          SPIR_ORDER_ROW_MAJOR, tol, positive_only);
            if (!positive_only) {
                integration_test<std::complex<double>, sparseir::Bosonic, sparseir::LogisticKernel, 1,
                               Eigen::RowMajor>(beta, wmax, epsilon, extra_dims, target_dim,
                                              SPIR_ORDER_ROW_MAJOR, tol, positive_only);
            }
        }

        // extra dims = {2,3,4}
        for (int target_dim = 0; target_dim < 4; ++target_dim) {
            std::vector<int> extra_dims = {2,3,4};
            std::cout << "Integration test for bosonic LogisticKernel, ColMajor, target_dim = " << target_dim << std::endl;
            integration_test<double, sparseir::Bosonic, sparseir::LogisticKernel, 4,
                           Eigen::ColMajor>(beta, wmax, epsilon, extra_dims, target_dim,
                                          SPIR_ORDER_COLUMN_MAJOR, tol, positive_only);
        }

        for (int target_dim = 0; target_dim < 4; ++target_dim) {
            std::vector<int> extra_dims = {2,3,4};
            std::cout << "Integration test for bosonic LogisticKernel, RowMajor, target_dim = " << target_dim << std::endl;
            integration_test<double, sparseir::Bosonic, sparseir::LogisticKernel, 4,
                           Eigen::RowMajor>(beta, wmax, epsilon, extra_dims, target_dim,
                                          SPIR_ORDER_ROW_MAJOR, tol, positive_only);
        }
    }
}