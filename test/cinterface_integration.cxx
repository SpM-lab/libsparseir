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

template <int ndim>
std::array<Eigen::Index, ndim>
_get_dims(int target_dim_size, const std::vector<int> &extra_dims,
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
    Eigen::VectorXd g_IR(basis_size * extra_size);
    status = spir_dlr_to_IR(dlr, order, ndim, &basis_size, target_dim,
                            coeffs.data(), g_IR.data());
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    spir_funcs *dlr_u;
    status = spir_dlr_get_u(dlr, &dlr_u);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    spir_funcs *ir_u = nullptr;
    status = spir_finite_temp_basis_get_u(basis, &ir_u);
    REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

    spir_destroy_finite_temp_basis(basis);
    spir_destroy_dlr(dlr);
    spir_destroy_funcs(dlr_u);
    spir_destroy_funcs(ir_u);
}

TEST_CASE("Integration Test", "[cinterface]")
{
    std::vector<int> extra_dims = {};
    integration_test<sparseir::Fermionic, sparseir::LogisticKernel, 1,
                     Eigen::ColMajor>(1.0, 10.0, 1e-10, extra_dims, 0,
                                      SPIR_ORDER_COLUMN_MAJOR);
}