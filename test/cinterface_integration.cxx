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

template <typename T, int ndim, Eigen::StorageOptions ORDER>
Eigen::Tensor<T, ndim, ORDER>
_evaluate_greens_function_batch(const Eigen::Tensor<T, ndim, ORDER> &coeffs,
                                const spir_funcs *u, int target_dim,
                                const Eigen::VectorXd &x_values)
{
    int32_t status;
    Eigen::Tensor<T, ndim, ORDER> coeffs_targetdim0 =
        sparseir::movedim(coeffs, target_dim, 0);

    int32_t funcs_size;
    status = spir_funcs_get_size(u, &funcs_size);
    _assert(status == SPIR_COMPUTATION_SUCCESS);

    // Calculate extra dimensions size
    Eigen::Index extra_size = 1;
    for (int i = 1; i < ndim; ++i) {
        extra_size *= coeffs_targetdim0.dimension(i);
    }

    // Create result tensor
    std::array<Eigen::Index, ndim> dims;
    dims[0] = x_values.size();
    for (int i = 1; i < ndim; ++i) {
        dims[i] = coeffs_targetdim0.dimension(i);
    }
    Eigen::Tensor<T, ndim, ORDER> g(dims);

    // Evaluate all basis functions at once
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER> u_eval_mat(x_values.size(), funcs_size);
    for (Eigen::Index i = 0; i < x_values.size(); ++i) {
        Eigen::VectorXd u_eval(funcs_size);
        status = spir_evaluate_funcs(u, x_values(i), u_eval.data());
        _assert(status == SPIR_COMPUTATION_SUCCESS);
        u_eval_mat.row(i) = u_eval.transpose().cast<T>();
    }

    // Map input tensors to matrices
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER>>
        coeffs_mat(coeffs_targetdim0.data(), funcs_size, extra_size);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER>>
        g_mat(g.data(), x_values.size(), extra_size);

    // Perform single matrix multiplication
    g_mat = u_eval_mat * coeffs_mat;

    // Move dimensions back to original order
    return sparseir::movedim(g, 0, target_dim);
}

template <typename T, int ndim, Eigen::StorageOptions ORDER>
Eigen::Tensor<T, ndim - 1, ORDER>
_evaluate_greens_function(const Eigen::Tensor<T, ndim, ORDER> &coeffs,
                          const spir_funcs *u, int target_dim, double x)
{
    int32_t status;
    Eigen::Tensor<T, ndim, ORDER> coeffs_targetdim0 =
        sparseir::movedim(coeffs, target_dim, 0);

    int32_t funcs_size;
    status = spir_funcs_get_size(u, &funcs_size);
    _assert(status == SPIR_COMPUTATION_SUCCESS);

    // Evaluate the basis functions at x
    Eigen::VectorXd u_eval(funcs_size);
    status = spir_evaluate_funcs(u, x, u_eval.data());
    _assert(status == SPIR_COMPUTATION_SUCCESS);

    // Calculate extra dimensions size
    Eigen::Index extra_size = 1;
    for (int i = 1; i < ndim; ++i) {
        extra_size *= coeffs_targetdim0.dimension(i);
    }

    // Create result tensor and map it to matrix
    Eigen::Tensor<T, ndim - 1, ORDER> g;
    std::array<Eigen::Index, ndim - 1> dims;
    for (int i = 0; i < ndim - 1; ++i) {
        dims[i] = coeffs_targetdim0.dimension(i + 1);
    }
    g.resize(dims);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER>> g_mat(
        g.data(), 1, extra_size);

    // Map input tensors to matrices and perform matrix multiplication
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER>>
        coeffs_mat(coeffs_targetdim0.data(), funcs_size, extra_size);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, ORDER> u_eval_mat =
        u_eval.transpose().cast<T>();
    g_mat = u_eval_mat * coeffs_mat;

    // Move dimensions back to original order
    return sparseir::movedim(g, 0, target_dim);
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

    // Sampling tau
    int32_t num_tau_points;
    status = spir_sampling_get_num_points(tau_sampling, &num_tau_points);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
    Eigen::VectorXd tau_points(num_tau_points);
    status = spir_sampling_get_tau_points(tau_sampling, tau_points.data());
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

    spir_funcs *dlr_u;
    status = spir_dlr_get_u(dlr, &dlr_u);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    spir_funcs *ir_u = nullptr;
    status = spir_finite_temp_basis_get_u(basis, &ir_u);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    Eigen::Tensor<double, ndim, ORDER> g_from_IR =
        _evaluate_greens_function_batch<double, ndim, ORDER>(g_IR, ir_u,
                                                             target_dim,
                                                             tau_points);
    Eigen::Tensor<double, ndim, ORDER> g_from_DLR =
        _evaluate_greens_function_batch<double, ndim, ORDER>(coeffs, dlr_u,
                                                       target_dim, tau_points);

    Eigen::Tensor<double, ndim, ORDER> diff = (g_from_IR - g_from_DLR).abs();

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> diff_vec(diff.data(), diff.size());
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> ref_vec(g_from_IR.data(), g_from_IR.size());
    double max_diff = diff_vec.maxCoeff();
    double max_ref = ref_vec.maxCoeff();

    REQUIRE(max_diff <= epsilon * max_ref);

    // Evaluate the Greens function at x
    for (auto x : tau_points) {
        Eigen::Tensor<double, ndim - 1, ORDER> g_from_IR =
            _evaluate_greens_function<double, ndim, ORDER>(g_IR, ir_u,
                                                           target_dim, x);
        Eigen::Tensor<double, ndim - 1, ORDER> g_from_DLR =
            _evaluate_greens_function<double, ndim, ORDER>(coeffs, dlr_u,
                                                           target_dim, x);

        // Eigen::Tensor<double, ndim - 1, ORDER> diff = g_from_IR - g_from_DLR;
        // double max_diff = diff.abs().maximum();
        // double max_ref = g_from_IR.abs().maximum();
        /// REQUIRE(max_diff <= epsilon * max_ref);
        // double diff = (g_from_IR - g_from_DLR).abs().maximum()()(0);

        // REQUIRE(g_from_IR(0) == Approx(g_from_DLR(0)).epsilon(epsilon));
        // std::cout << "x: " << x << std::endl;
        // std::cout << "g_from_IR: " << g_from_IR << std::endl;
        // std::cout << "g_from_DLR: " << g_from_DLR << std::endl;
        // std::cout << "diff: " << (g_from_IR - g_from_DLR).abs().maximum() <<
        // std::endl;
    }
    // Eigen::Tensor<double, ndim, ORDER> g = _evaluate_greens_function(coeffs,
    // ir_u, target_dim, x);

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