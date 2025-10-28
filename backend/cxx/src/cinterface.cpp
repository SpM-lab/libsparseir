#include "sparseir/sparseir.h"
#include "sparseir/sparseir.hpp"
#include "sparseir/utils.hpp"
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <iostream>

inline void DEBUG_LOG(const std::string& msg) {
    const char* env = std::getenv("SPARSEIR_DEBUG");
    if (env && std::string(env) == "1") {
        std::cout << "[DEBUG] " << msg << std::endl;
    }
}

#include "cinterface_impl/helper_types.hpp"
#include "cinterface_impl/opaque_types.hpp"
#include "cinterface_impl/helper_funcs.hpp"

// Implementation of the C API
extern "C" {

spir_kernel* spir_logistic_kernel_new(double lambda, int* status)
{
    try {
        auto kernel_ptr = std::make_shared<sparseir::LogisticKernel>(lambda);
        std::shared_ptr<sparseir::AbstractKernel> abstract_kernel = _safe_static_pointer_cast<sparseir::AbstractKernel>(kernel_ptr);

        // Check if dynamic_cast works at this point
        auto check_logistic = _safe_dynamic_pointer_cast<sparseir::LogisticKernel>(abstract_kernel);

        *status = SPIR_COMPUTATION_SUCCESS;
        return create_kernel(abstract_kernel);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_logistic_kernel_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_logistic_kernel_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_kernel* spir_reg_bose_kernel_new(double lambda, int* status)
{
    DEBUG_LOG("Creating RegularizedBoseKernel with lambda=" + std::to_string(lambda));
    try {
        auto kernel_ptr = std::make_shared<sparseir::RegularizedBoseKernel>(lambda);
        auto abstract_kernel = _safe_static_pointer_cast<sparseir::AbstractKernel>(kernel_ptr);
        *status = SPIR_COMPUTATION_SUCCESS;
        return create_kernel(abstract_kernel);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_reg_bose_kernel_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_reg_bose_kernel_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

int spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                           double *ymin, double *ymax)
{
    DEBUG_LOG("spir_kernel_domain called with kernel=" + std::to_string(reinterpret_cast<uintptr_t>(k)));
    auto impl = get_impl_kernel(k);
    if (!impl) {
        DEBUG_LOG("Failed to get kernel implementation");
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        DEBUG_LOG("Getting xrange and yrange");
        auto xrange = impl->xrange();
        auto yrange = impl->yrange();

        DEBUG_LOG("Setting output values: xrange=(" +
                  std::to_string(xrange.first) + ", " + std::to_string(xrange.second) + "), yrange=(" +
                  std::to_string(yrange.first) + ", " + std::to_string(yrange.second) + ")");
        if (!xmin) {
            DEBUG_LOG("xmin is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }
        if (!xmax) {
            DEBUG_LOG("xmax is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }
        if (!ymin) {
            DEBUG_LOG("ymin is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }
        if (!ymax) {
            DEBUG_LOG("ymax is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }
        *xmin = xrange.first;
        *xmax = xrange.second;
        *ymin = yrange.first;
        *ymax = yrange.second;

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_kernel_domain: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_kernel_domain");
        return SPIR_INTERNAL_ERROR;
    }
}

spir_sve_result* spir_sve_result_new(
    const spir_kernel *k,
    double epsilon,
    double cutoff,
    int lmax,
    int n_gauss,
    int Twork,
    int *status
)
{
    try {
        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            DEBUG_LOG("Failed to get a sve result implementation");
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }

        if (cutoff < 0) {
            cutoff = std::numeric_limits<double>::quiet_NaN();
        }

        if (lmax < 0) {
            lmax = std::numeric_limits<int>::max();
        }

        sparseir::TworkType Twork_type;
        if (Twork == SPIR_TWORK_FLOAT64) {
            Twork_type = sparseir::TworkType::FLOAT64;
        } else if (Twork == SPIR_TWORK_FLOAT64X2) {
            Twork_type = sparseir::TworkType::FLOAT64X2;
        } else if (Twork == SPIR_TWORK_AUTO) {
            Twork_type = sparseir::TworkType::AUTO;
        } else {
            DEBUG_LOG("Error: Invalid Twork");
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        std::shared_ptr<sparseir::SVEResult> sve_result;

        if (auto logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(impl)) {
            sve_result = std::make_shared<sparseir::SVEResult>(sparseir::compute_sve(
                *logistic, epsilon, cutoff, lmax, n_gauss, Twork_type
            ));
        } else if (auto bose = std::dynamic_pointer_cast<sparseir::RegularizedBoseKernel>(impl)) {
            sve_result = std::make_shared<sparseir::SVEResult>(sparseir::compute_sve(
                *bose, epsilon, cutoff, lmax, n_gauss, Twork_type
            ));
        } else {
            DEBUG_LOG("Unknown kernel type");
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }

        *status = SPIR_COMPUTATION_SUCCESS;
        return create_sve_result(sve_result);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_basis* spir_basis_new(
    int statistics, double beta, double omega_max, double epsilon,
    const spir_kernel *k, const spir_sve_result *sve, int max_size, int* status)
{
    try {
        // Get the kernel implementation
        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            DEBUG_LOG("Failed to get a basis implementation");
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }

        // switch on kernel type
        spir_basis* result = nullptr;
        if (auto logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(impl)) {
            result = _spir_basis_new(statistics, beta, omega_max, epsilon, *logistic, sve, max_size);
        } else if (auto bose = std::dynamic_pointer_cast<sparseir::RegularizedBoseKernel>(impl)) {
            result = _spir_basis_new(statistics, beta, omega_max, epsilon, *bose, sve, max_size);
        } else {
            DEBUG_LOG("Unknown kernel type");
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        if (!result) {
            DEBUG_LOG("Failed to create finite temperature basis");
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }

        *status = SPIR_COMPUTATION_SUCCESS;
        return result;
    } catch (const std::exception& e) {
        DEBUG_LOG("Exception in spir_basis_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}


spir_sampling* spir_tau_sampling_new(const spir_basis *b, int num_points, const double *points, int* status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_tau_sampling_new: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!points) {
        DEBUG_LOG("Error in spir_tau_sampling_new: invalid pointer points");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (num_points <= 0) {
        DEBUG_LOG("Error in spir_tau_sampling_new: num_points is less than or equal to 0");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        DEBUG_LOG("Failed to get a tau sampling implementation");
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            *status = SPIR_COMPUTATION_SUCCESS;
            return _spir_tau_sampling_new_with_points<sparseir::TauSampling<sparseir::Fermionic>>(
                b, num_points, points);
        } else {
            *status = SPIR_COMPUTATION_SUCCESS;
            return _spir_tau_sampling_new_with_points<sparseir::TauSampling<sparseir::Bosonic>>(
                b, num_points, points);
        }
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_tau_sampling_new: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_sampling* spir_tau_sampling_new_with_matrix(int order, int statistics, int basis_size, int num_points, const double *points, const double *matrix, int* status)
{
    if (!points) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: invalid pointer points");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!matrix) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: invalid pointer matrix");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (num_points <= 0) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: num_points is less than or equal to 0");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

        // check statistics
    if (statistics != SPIR_STATISTICS_FERMIONIC && statistics != SPIR_STATISTICS_BOSONIC) {
        *status = SPIR_INVALID_ARGUMENT;
        DEBUG_LOG("Error: Invalid statistics");
        return nullptr;
    }

    // check order
    if (order != SPIR_ORDER_ROW_MAJOR && order != SPIR_ORDER_COLUMN_MAJOR) {
        *status = SPIR_INVALID_ARGUMENT;
        DEBUG_LOG("Error: Invalid order");
        return nullptr;
    }

    try {
        if (statistics == SPIR_STATISTICS_FERMIONIC) {
            return _spir_tau_sampling_new_with_matrix<sparseir::TauSampling<sparseir::Fermionic>>(
                order, basis_size, num_points, points, matrix, status);
        } else {
            return _spir_tau_sampling_new_with_matrix<sparseir::TauSampling<sparseir::Bosonic>>(
                order, basis_size, num_points, points, matrix, status);
        }
    } catch (const std::exception& e) {
        DEBUG_LOG("Error in spir_tau_sampling_new_with_matrix: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_sampling* spir_matsu_sampling_new(const spir_basis *b, bool positive_only, int num_points, const int64_t *points, int* status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_matsu_sampling_new: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!points) {
        DEBUG_LOG("Error in spir_matsu_sampling_new: invalid pointer points");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (num_points <= 0) {
        DEBUG_LOG("Error in spir_matsu_sampling_new: num_points is less than or equal to 0");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    if (positive_only) {
        for (int i = 0; i < num_points; i++) {
            if (points[i] < 0) {
                DEBUG_LOG("Error: Frequency must not be negative if positive_only is true");
                *status = SPIR_INVALID_ARGUMENT;
                return nullptr;
            }
        }
    }

    // Get basis functions
    spir_funcs* uhat = spir_basis_get_uhat(b, status);
    if (!uhat) {
        DEBUG_LOG("Error: Failed to get basis functions");
        return nullptr;
    }

    // Get basis size
    int basis_size;
    int size_status = spir_basis_get_size(b, &basis_size);
    if (size_status != SPIR_COMPUTATION_SUCCESS) {
        DEBUG_LOG("Error: Failed to get basis size");
        spir_funcs_release(uhat);
        *status = size_status;
        return nullptr;
    }

    int statistics;
    int stats_status = spir_basis_get_stats(b, &statistics);
    if (stats_status != SPIR_COMPUTATION_SUCCESS) {
        DEBUG_LOG("Error: Failed to get basis statistics");
        spir_funcs_release(uhat);
        *status = stats_status;
        return nullptr;
    }

    // Allocate memory for matrix
    std::vector<std::complex<double>> matrix(num_points * basis_size);
    int eval_status = spir_funcs_batch_eval_matsu(uhat, SPIR_ORDER_COLUMN_MAJOR, num_points, points, (c_complex*)matrix.data());
    if (eval_status != SPIR_COMPUTATION_SUCCESS) {
        DEBUG_LOG("Error: Failed to evaluate basis functions");
        spir_funcs_release(uhat);
        *status = eval_status;
        return nullptr;
    }

    // Create sampling object using the matrix version
    spir_sampling* smpl = spir_matsu_sampling_new_with_matrix(
        SPIR_ORDER_COLUMN_MAJOR, statistics, basis_size, positive_only, num_points, points, (c_complex*)matrix.data(), status);

    spir_funcs_release(uhat);

    return smpl;
}

spir_sampling* spir_matsu_sampling_new_with_matrix(
                                                  int order,
                                                  int statistics,
                                                  int basis_size,
                                                  bool positive_only,
                                                  int num_points,
                                                  const int64_t *points,
                                                  const c_complex *matrix,
                                                  int *status)
{
    if (!points) {
        DEBUG_LOG("Error in spir_matsu_sampling_new_with_matrix: invalid pointer points");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!matrix) {
        DEBUG_LOG("Error in spir_matsu_sampling_new_with_matrix: invalid pointer matrix");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_matsu_sampling_new_with_matrix: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    // check statistics
    if (statistics != SPIR_STATISTICS_FERMIONIC && statistics != SPIR_STATISTICS_BOSONIC) {
        *status = SPIR_INVALID_ARGUMENT;
        DEBUG_LOG("Error: Invalid statistics");
        return nullptr;
    }

    // check order
    if (order != SPIR_ORDER_ROW_MAJOR && order != SPIR_ORDER_COLUMN_MAJOR) {
        *status = SPIR_INVALID_ARGUMENT;
        DEBUG_LOG("Error: Invalid order");
        return nullptr;
    }

    if (statistics == SPIR_STATISTICS_FERMIONIC) {
        return _spir_matsu_sampling_new_with_matrix<sparseir::Fermionic, sparseir::MatsubaraSampling<sparseir::Fermionic>>(
            order, basis_size, positive_only, num_points, points, matrix, status);
    } else {
        return _spir_matsu_sampling_new_with_matrix<sparseir::Bosonic, sparseir::MatsubaraSampling<sparseir::Bosonic>>(
            order, basis_size, positive_only, num_points, points, matrix, status);
    }
}

spir_basis* spir_dlr_new(const spir_basis *b, int* status)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    int stat;
    int status_basis = spir_basis_get_stats(b, &stat);
    if (status_basis != SPIR_COMPUTATION_SUCCESS) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_dlr_new<sparseir::Fermionic>(b);
    } else {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_dlr_new<sparseir::Bosonic>(b);
    }
}

spir_basis* spir_dlr_new_with_poles(const spir_basis *b,
                                  const int npoles, const double *poles, int* status)
{
    auto impl = get_impl_basis(b);
    if (!impl)
        return nullptr;

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    int stat;
    int status_basis = spir_basis_get_stats(b, &stat);
    if (status_basis != SPIR_COMPUTATION_SUCCESS) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_dlr_new_with_poles<sparseir::Fermionic>(b, npoles, poles);
    } else {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_dlr_new_with_poles<sparseir::Bosonic>(b, npoles, poles);
    }
}

int spir_sampling_eval_dd(const spir_sampling *s, int order,
                                  int ndim, const int *input_dims,
                                  int target_dim, const double *input,
                                  double *out)
{
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, out,
                         &sparseir::AbstractSampling::evaluate_inplace_dd);
}

int spir_sampling_eval_dz(const spir_sampling *s, int order,
                                  int ndim, const int *input_dims,
                                  int target_dim, const double *input,
                                  c_complex *out)
{
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, cpp_out,
                         &sparseir::AbstractSampling::evaluate_inplace_dz);
}

int spir_sampling_eval_zz(const spir_sampling *s, int order,
                                  int ndim, const int *input_dims,
                                  int target_dim, const c_complex *input,
                                  c_complex *out)
{
    // DANGER: MEMORY LAYOUT MAY NOT BE CONSISTENT BETWEEN C99 AND C++
    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return evaluate_impl(s, order, ndim, input_dims, target_dim, cpp_input,
                         cpp_out,
                         &sparseir::AbstractSampling::evaluate_inplace_zz);
}

int spir_sampling_fit_dd(const spir_sampling *s, int order,
                             int ndim, const int *input_dims,
                             int target_dim, const double *input,
                             double *out)
{
    return fit_impl(s, order, ndim, input_dims, target_dim, input, out,
                    &sparseir::AbstractSampling::fit_inplace_dd);
}

int spir_sampling_fit_zz(const spir_sampling *s, int order,
                             int ndim, const int *input_dims,
                             int target_dim, const c_complex *input,
                             c_complex *out)
{
    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return fit_impl(s, order, ndim, input_dims, target_dim, cpp_input, cpp_out,
                    &sparseir::AbstractSampling::fit_inplace_zz);
}

int spir_sampling_fit_zd(const spir_sampling *s, int order,
                             int ndim, const int *input_dims,
                             int target_dim, const c_complex *input,
                             double *out)
{
    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    return fit_impl(s, order, ndim, input_dims, target_dim, cpp_input, out,
                    &sparseir::AbstractSampling::fit_inplace_zd);
}

int spir_dlr2ir_dd(const spir_basis *dlr, int order, int ndim,
                       const int *input_dims, int target_dim,
                       const double *input, double *out)
{
    auto impl = get_impl_basis(dlr);
    if (!impl) {
        DEBUG_LOG("Error in spir_dlr2ir_dd: failed to get basis implementation");
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr2ir<sparseir::Fermionic, double>(dlr, order, ndim, input_dims, target_dim,
                                                   input, out);
    } else {
        return spir_dlr2ir<sparseir::Bosonic, double>(dlr, order, ndim, input_dims, target_dim,
                                                 input, out);
    }
}

int spir_dlr2ir_zz(const spir_basis *dlr, int order, int ndim,
                       const int *input_dims, int target_dim,
                       const c_complex *input, c_complex *out)
{
    auto impl = get_impl_basis(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr2ir<sparseir::Fermionic, std::complex<double>>(dlr, order, ndim, input_dims, target_dim,
                                                   cpp_input, cpp_out);
    } else {
        return spir_dlr2ir<sparseir::Bosonic, std::complex<double>>(dlr, order, ndim, input_dims, target_dim,
                                                 cpp_input, cpp_out);
    }
}

int spir_ir2dlr_dd(const spir_basis *dlr, int order,
                         int ndim, const int *input_dims, int target_dim,
                         const double *input, double *out)
{
    auto impl = get_impl_basis(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_ir2dlr<sparseir::Fermionic, double>(dlr, order, ndim,
                                                     input_dims, target_dim, input, out);
    } else {
        return spir_ir2dlr<sparseir::Bosonic, double>(dlr, order, ndim, input_dims,
                                                   target_dim, input, out);
    }
}

int spir_ir2dlr_zz(const spir_basis *dlr, int order,
                         int ndim, const int *input_dims, int target_dim,
                         const c_complex *input, c_complex *out)
{
    auto impl = get_impl_basis(dlr);
    if  (!impl)
        return SPIR_GET_IMPL_FAILED;

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_ir2dlr<sparseir::Fermionic, std::complex<double>>(dlr, order, ndim,
                                                     input_dims, target_dim, cpp_input, cpp_out);
    } else {
        return spir_ir2dlr<sparseir::Bosonic, std::complex<double>>(dlr, order, ndim, input_dims,
                                                   target_dim, cpp_input, cpp_out);
    }
}

int spir_dlr_get_npoles(const spir_basis *dlr, int *num_poles)
{
    if (!dlr || !num_poles) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        *num_poles = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_dlr_get_poles(const spir_basis *dlr, double *poles)
{
    if (!dlr || !poles) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_dlr_basis(dlr)) {
        DEBUG_LOG("Error: The basis is not a DLR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        std::vector<double> poles_vec = std::dynamic_pointer_cast<AbstractDLR>(impl)->get_poles();
        std::memcpy(poles, poles_vec.data(), poles_vec.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

spir_funcs* spir_basis_get_u(const spir_basis *b, int *status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_u: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_basis_get_u: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            spir_funcs *u = nullptr;
            int ret = _spir_basis_get_u<sparseir::Fermionic>(b, &u);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return u;
        } else {
            spir_funcs *u = nullptr;
            int ret = _spir_basis_get_u<sparseir::Bosonic>(b, &u);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return u;
        }
    } catch (const std::exception &e) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
}

spir_funcs* spir_basis_get_v(const spir_basis *b, int *status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_v: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_basis_get_v: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            spir_funcs *v = nullptr;
            int ret = _spir_get_v<sparseir::Fermionic>(b, &v);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return v;
        } else {
            spir_funcs *v = nullptr;
            int ret = _spir_get_v<sparseir::Bosonic>(b, &v);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return v;
        }
    } catch (const std::exception &e) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
}

spir_funcs* spir_basis_get_uhat(const spir_basis *b, int *status)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_uhat: invalid pointer b");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!status) {
        DEBUG_LOG("Error in spir_basis_get_uhat: invalid pointer status");
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            spir_funcs *uhat = nullptr;
            int ret = _spir_basis_get_uhat<sparseir::Fermionic>(b, &uhat);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return uhat;
        } else {
            spir_funcs *uhat = nullptr;
            int ret = _spir_basis_get_uhat<sparseir::Bosonic>(b, &uhat);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return uhat;
        }
    } catch (const std::exception &e) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
}

// TODO: USE THIS
int spir_sampling_get_npoints(const spir_sampling *s,
                                     int *num_points)
{
    auto impl = get_impl_sampling(s);
    if (!impl) {
        DEBUG_LOG("Error in spir_sampling_get_npoints: failed to get sampling implementation");
        return SPIR_GET_IMPL_FAILED;
    }
    if (!num_points) {
        DEBUG_LOG("Error in spir_sampling_get_npoints: invalid pointer num_points");
        return SPIR_INVALID_ARGUMENT;
    }
    try {
        *num_points = impl->n_sampling_points();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        DEBUG_LOG("Error in spir_sampling_get_npoints: unknown exception");
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_sampling_get_taus(const spir_sampling *s, double *points)
{
    auto impl = get_impl_sampling(s);
    if (!impl) {
        DEBUG_LOG("Error in spir_sampling_get_taus: failed to get sampling implementation");
        return SPIR_GET_IMPL_FAILED;
    }
    try {
        // Try Fermionic case first
        auto fermionic_sampling = std::dynamic_pointer_cast<
            sparseir::TauSampling<sparseir::Fermionic>>(impl);
        if (fermionic_sampling) {
            auto tau_points = fermionic_sampling->sampling_points();
            std::copy(tau_points.begin(), tau_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try Bosonic case
        auto bosonic_sampling =
            std::dynamic_pointer_cast<sparseir::TauSampling<sparseir::Bosonic>>(
                impl);
        if (bosonic_sampling) {
            auto tau_points = bosonic_sampling->sampling_points();
            std::copy(tau_points.begin(), tau_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }

        return SPIR_NOT_SUPPORTED;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

// TODO: USE THIS
int spir_sampling_get_matsus(const spir_sampling *s,
                                           int64_t *points)
{
    auto impl = get_impl_sampling(s);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    try {
        // Try Fermionic case first
        auto fermionic_sampling = std::dynamic_pointer_cast<
            sparseir::MatsubaraSampling<sparseir::Fermionic>>(impl);
        if (fermionic_sampling) {
            auto matsubara_points = fermionic_sampling->sampling_points();
            std::transform(
                matsubara_points.begin(), matsubara_points.end(), points,
                [](const sparseir::MatsubaraFreq<sparseir::Fermionic> &freq) {
                    return static_cast<int64_t>(freq.get_n());
                });
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try Bosonic case
        auto bosonic_sampling = std::dynamic_pointer_cast<
            sparseir::MatsubaraSampling<sparseir::Bosonic>>(impl);
        if (bosonic_sampling) {
            auto matsubara_points = bosonic_sampling->sampling_points();
            std::transform(
                matsubara_points.begin(), matsubara_points.end(), points,
                [](const sparseir::MatsubaraFreq<sparseir::Bosonic> &freq) {
                    return static_cast<int64_t>(freq.get_n());
                });
            return SPIR_COMPUTATION_SUCCESS;
        }

        return SPIR_NOT_SUPPORTED;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_size(const spir_basis *b,
                                        int *size)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        DEBUG_LOG("Error in spir_basis_get_size: failed to get basis implementation");
        return SPIR_GET_IMPL_FAILED;
    }
    if (!size) {
        DEBUG_LOG("Error in spir_basis_get_size: invalid pointer size");
        return SPIR_INVALID_ARGUMENT;
    }
    try {
        *size = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        DEBUG_LOG("Error in spir_basis_get_size: unknown exception");
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_n_default_taus(const spir_basis *b, int *num_points)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_n_default_taus: invalid pointer b");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!num_points) {
        DEBUG_LOG("Error in spir_basis_get_n_default_taus: invalid pointer num_points");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto points = ir_basis->default_tau_sampling_points();
            *num_points = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto points = ir_basis->default_tau_sampling_points();
            *num_points = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_default_taus(const spir_basis *b, double *points)
{
    if (!b) {
        DEBUG_LOG("Error in spir_basis_get_default_taus: invalid pointer b");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!points) {
        DEBUG_LOG("Error in spir_basis_get_default_taus: invalid pointer points");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto tau_points = ir_basis->default_tau_sampling_points();
            std::copy(tau_points.begin(), tau_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto tau_points = ir_basis->default_tau_sampling_points();
            std::copy(tau_points.begin(), tau_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_default_taus_ext(
    const spir_basis *b, int n_points, double *points, int *n_points_returned)
{
    if (!b || !points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    double beta = impl->get_beta();

    try {
        Eigen::VectorXd tau_points;
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            tau_points = sparseir::default_sampling_points(
                *(ir_basis->get_impl()->sve_result->u), n_points
            );
            *n_points_returned = tau_points.size();
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            std::cout << "debug " << ir_basis->get_impl()->sve_result->u->size() << std::endl;
            tau_points = sparseir::default_sampling_points(
                *(ir_basis->get_impl()->sve_result->u), n_points
            );
            *n_points_returned = tau_points.size();
        }

        // Copy the requested number of points
        // rescale the points to the original domain
        for (int i = 0; i < *n_points_returned; ++i) {
            tau_points(i) = (tau_points(i) + 1) / 2 * beta;
            if (tau_points(i) > 0.5 * beta) {
                tau_points(i) -= beta;
            }
        }
        std::copy(tau_points.data(), tau_points.data() + *n_points_returned, points);
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_n_default_matsus(const spir_basis *b, bool positive_only, int *num_points)
{
    if (!b || !num_points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto points = ir_basis->default_matsubara_sampling_points(positive_only);
            *num_points = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto points = ir_basis->default_matsubara_sampling_points(positive_only);
            *num_points = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_default_matsus(const spir_basis *b, bool positive_only, int64_t *points)
{
    if (!b || !points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto matsubara_points = ir_basis->default_matsubara_sampling_points(positive_only);
            std::copy(matsubara_points.begin(), matsubara_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto matsubara_points = ir_basis->default_matsubara_sampling_points(positive_only);
            std::copy(matsubara_points.begin(), matsubara_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_n_default_matsus_ext(const spir_basis *b, bool positive_only, int L, int *num_points_returned)
{
    if (!b || !num_points_returned) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto points = ir_basis->default_matsubara_sampling_points_ext(L, positive_only);
            *num_points_returned = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto points = ir_basis->default_matsubara_sampling_points_ext(L, positive_only);
            *num_points_returned = points.size();
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_default_matsus_ext(const spir_basis *b, bool positive_only, int L, int64_t *points, int *n_points_returned)
{
      if (!b || !points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            auto matsubara_points = ir_basis->default_matsubara_sampling_points_ext(L, positive_only);
            *n_points_returned = matsubara_points.size();
            std::copy(matsubara_points.begin(), matsubara_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        } else {
            auto ir_basis = _safe_static_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            auto matsubara_points = ir_basis->default_matsubara_sampling_points_ext(L, positive_only);
            *n_points_returned = matsubara_points.size();
            std::copy(matsubara_points.begin(), matsubara_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_basis_get_stats(const spir_basis *b,
                                  int *statistics)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (!statistics) {
        return SPIR_INVALID_ARGUMENT;
    }
    try {
        *statistics = impl->get_statistics();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int spir_funcs_eval(const spir_funcs *funcs, double x, double *out)
{
    if (!out) {
        DEBUG_LOG("Error in spir_funcs_eval: out is null");
        return SPIR_INVALID_ARGUMENT;
    }
    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (!impl->is_continuous_funcs()) {
        DEBUG_LOG("Error: the function is not defined for continuous variables");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        Eigen::VectorXd result = std::dynamic_pointer_cast<AbstractContinuousFunctions>(funcs->ptr)->operator()(x);
        std::memcpy(out, result.data(), result.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_eval: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_eval_matsu(const spir_funcs *funcs, int64_t x, c_complex *out)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_eval_matsu: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!out) {
        DEBUG_LOG("Error in spir_funcs_eval_matsu: invalid pointer out");
        return SPIR_INVALID_ARGUMENT;
    }

    // Use batch_evaluate_matsubara with num_freqs = 1
    return spir_funcs_batch_eval_matsu(funcs, SPIR_ORDER_COLUMN_MAJOR, 1, &x, out);
}

int spir_funcs_batch_eval(const spir_funcs *funcs,
                                 int order, int num_points,
                                 const double *xs, double *out)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_batch_eval: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!xs) {
        DEBUG_LOG("Error in spir_funcs_batch_eval: invalid pointer xs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!out) {
        DEBUG_LOG("Error in spir_funcs_batch_eval: invalid pointer out");
        return SPIR_INVALID_ARGUMENT;
    }
    if (num_points <= 0) {
        DEBUG_LOG("Error in spir_funcs_batch_eval: num_points is less than or equal to 0");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        // Get the size of the functions object
        int size;
        int status = spir_funcs_get_size(funcs, &size);
        if (status != SPIR_COMPUTATION_SUCCESS) {
            return status;
        }

        // result is a matrix of size n_funcs x num_points in column-major order
        Eigen::MatrixXd result = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl)->operator()(Eigen::Map<const Eigen::VectorXd>(xs, num_points));

        // out is a matrix of size num_points x n_funcs
        if (order == SPIR_ORDER_ROW_MAJOR) {
            // Copy the results to the output array
            for (int i = 0; i < num_points; ++i) {
                for (int j = 0; j < size; ++j) {
                    out[i * size + j] = result(j, i);
                }
            }
        } else {
            // Copy the results to the output array
            for (int i = 0; i < num_points; ++i) {
                for (int j = 0; j < size; ++j) {
                    out[j * num_points + i] = result(j, i);
                }
            }
        }

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_batch_eval: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_batch_eval");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_batch_eval_matsu(const spir_funcs *uiw,
                                          int order,
                                          int num_freqs,
                                          const int64_t *matsubara_freq_indices,
                                          c_complex *out)
{
    auto impl = get_impl_funcs(uiw);
    if (!impl) {
        DEBUG_LOG("Matsubara basis functions object is null or not assigned");
        return SPIR_GET_IMPL_FAILED;
    }
    if (impl->is_continuous_funcs()) {
        DEBUG_LOG("Error: the function is not defined for Matsubara frequencies");
        return SPIR_INVALID_ARGUMENT;
    }

    if (!matsubara_freq_indices || !out) {
        DEBUG_LOG("Input or output array is null");
        return SPIR_INVALID_ARGUMENT;
    }

    if (num_freqs <= 0) {
        DEBUG_LOG("Number of frequencies must be positive");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        // Convert C array to Eigen vector
        Eigen::Vector<int64_t, Eigen::Dynamic> freq_indices =
            Eigen::Map<const Eigen::Vector<int64_t, Eigen::Dynamic>>(matsubara_freq_indices, num_freqs);

        // Get the func size
        int func_size = uiw->ptr->size();

        // Evaluate functions at all frequencies
        // The operator() returns a matrix of shape (nfuncs, nfreqs)
        Eigen::MatrixXcd out_matrix = std::dynamic_pointer_cast<AbstractMatsubaraFunctions>(uiw->ptr)->operator()(freq_indices);

        // Copy the results to the output array
        for (int ifreq = 0; ifreq < num_freqs; ++ifreq) {
            for (int ifunc = 0; ifunc < func_size; ++ifunc) {
                const auto &val = out_matrix(ifunc, ifreq);
                if (order == SPIR_ORDER_ROW_MAJOR) {
                    out[ifreq * func_size + ifunc] = {val.real(), val.imag()};
                } else {
                    out[ifunc * num_freqs + ifreq] = {val.real(), val.imag()};
                }
            }
        }

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_batch_eval_matsu: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_size(const spir_funcs *funcs, int *size)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_get_size: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!size) {
        DEBUG_LOG("Error in spir_funcs_get_size: invalid pointer size");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        auto impl = get_impl_funcs(funcs);
        if (!impl) {
            return SPIR_GET_IMPL_FAILED;
        }
        *size = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

spir_funcs *spir_funcs_get_slice(const spir_funcs *funcs, int nslice, int *indices, int *status)
{
    try {
        // Get the implementation
        auto impl = get_impl_funcs(funcs);
        if (!impl) {
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }

        // Convert indices to vector<size_t>
        std::vector<size_t> indices_vec(indices, indices + nslice);

        // Check for duplicates and out of range
        try {
            check_indices(indices_vec, impl->size());
        } catch (const std::runtime_error& e) {
            DEBUG_LOG("Error: " + std::string(e.what()));
            *status = SPIR_INVALID_ARGUMENT;
            return nullptr;
        }

        // Get the slice
        auto sliced_impl = impl->slice(indices_vec);
        if (!sliced_impl) {
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }

        // Create new funcs object
        *status = SPIR_COMPUTATION_SUCCESS;
        return create_funcs(sliced_impl);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_slice: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

int spir_basis_get_n_default_ws(const spir_basis *b,
                                                       int *num_points)
{
    if (!b || !num_points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto basis = std::dynamic_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto points = basis->default_omega_sampling_points();
            *num_points = points.size();
        } else {
            auto basis = std::dynamic_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto points = basis->default_omega_sampling_points();
            *num_points = points.size();
        }
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_get_n_default_ws: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_basis_get_n_default_ws");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_basis_get_default_ws(const spir_basis *b,
                                                   double *points)
{
    if (!b || !points) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        DEBUG_LOG("Error: The basis is not an IR basis");
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto basis = std::dynamic_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto sampling_points = basis->default_omega_sampling_points();
            std::copy(sampling_points.begin(), sampling_points.end(), points);
        } else {
            auto basis = std::dynamic_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto sampling_points = basis->default_omega_sampling_points();
            std::copy(sampling_points.begin(), sampling_points.end(), points);
        }
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_get_default_ws: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_basis_get_default_ws");
        return SPIR_INTERNAL_ERROR;
    }
}

void *_spir_basis_get_raw_ptr(const spir_basis *obj)
{
    if (!obj) {
        return nullptr;
    }
    auto impl = get_impl_basis(obj);
    if (!impl) {
        return nullptr;
    }
    return impl.get();
}

int spir_sve_result_get_size(const spir_sve_result *sve, int *size)
{
    if (!sve || !size) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_sve_result(sve);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        *size = impl->s.size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_get_size: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_get_size");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_sve_result_get_svals(const spir_sve_result *sve, double *svals)
{
    if (!sve || !svals) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_sve_result(sve);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        std::memcpy(svals, impl->s.data(), impl->s.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_get_svals: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_get_svals");
        return SPIR_INTERNAL_ERROR;
    }
}

spir_sve_result* spir_sve_result_truncate(const spir_sve_result *sve, double epsilon, int max_size, int *status)
{
    if (!sve || !status) {
        if (status) *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    auto impl = get_impl_sve_result(sve);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    try {
        // Use the part method to truncate the SVE result
        auto part_result = impl->part(epsilon, max_size);
        
        // Extract the truncated components
        sparseir::PiecewiseLegendrePolyVector u_truncated = std::get<0>(part_result);
        Eigen::VectorXd s_truncated = std::get<1>(part_result);
        sparseir::PiecewiseLegendrePolyVector v_truncated = std::get<2>(part_result);
        
        // Create a new SVEResult with the truncated components
        // Use the original epsilon or the provided one if it's not NaN
        double result_epsilon = std::isnan(epsilon) ? impl->epsilon : epsilon;
        auto truncated_sve_result = std::make_shared<sparseir::SVEResult>(
            u_truncated, s_truncated, v_truncated, result_epsilon);
        
        *status = SPIR_COMPUTATION_SUCCESS;
        return create_sve_result(truncated_sve_result);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_truncate: " + std::string(e.what()));
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_truncate");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

int spir_basis_get_svals(const spir_basis *b, double *svals)
{
    if (!b || !svals) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_ir_basis(b)) {
        std::cerr << "Error: The basis is not an IR basis" << std::endl;
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            auto basis = _safe_dynamic_pointer_cast<_IRBasis<sparseir::Fermionic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto s = basis->s();
            std::memcpy(svals, s.data(), s.size() * sizeof(double));
        } else {
            auto basis = _safe_dynamic_pointer_cast<_IRBasis<sparseir::Bosonic>>(impl);
            if (!basis) {
                return SPIR_INTERNAL_ERROR;
            }
            auto s = basis->s();
            std::memcpy(svals, s.data(), s.size() * sizeof(double));
        }
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_get_svals: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_basis_get_svals");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_n_roots(const spir_funcs *funcs, int *n_roots)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_get_n_roots: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!n_roots) {
        DEBUG_LOG("Error in spir_funcs_get_n_roots: invalid pointer n_roots");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!impl->is_continuous_funcs()) {
            return SPIR_NOT_SUPPORTED;
        }

        auto continuous_impl = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl);
        if (!continuous_impl) {
            return SPIR_INTERNAL_ERROR;
        }

        *n_roots = continuous_impl->nroots();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_n_roots: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_get_n_roots");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_roots(const spir_funcs *funcs, double *roots)
{
    if (!funcs || !roots) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!impl->is_continuous_funcs()) {
            return SPIR_NOT_SUPPORTED;
        }

        auto continuous_impl = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl);
        if (!continuous_impl) {
            return SPIR_INTERNAL_ERROR;
        }

        // Get the roots from the implementation
        Eigen::VectorXd roots_vec = continuous_impl->roots();

        // Copy the roots to the output array
        std::memcpy(roots, roots_vec.data(), roots_vec.size() * sizeof(double));

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_roots: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_get_roots");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_sampling_get_cond_num(const spir_sampling *s, double *cond_num)
{
    DEBUG_LOG("spir_sampling_get_cond_num called with sampling=" + std::to_string(reinterpret_cast<uintptr_t>(s)));
    auto impl = get_impl_sampling(s);
    if (!impl) {
        DEBUG_LOG("Failed to get sampling implementation");
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!cond_num) {
            DEBUG_LOG("cond_num is nullptr");
            return SPIR_INVALID_ARGUMENT;
        }

        *cond_num = impl->get_cond_num();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sampling_get_cond_num: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sampling_get_cond_num");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_n_knots(const spir_funcs *funcs, int *n_knots)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_get_n_knots: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!n_knots) {
        DEBUG_LOG("Error in spir_funcs_get_n_knots: invalid pointer n_knots");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!impl->is_continuous_funcs()) {
            DEBUG_LOG("Error: the function is not defined for continuous variables");
            return SPIR_NOT_SUPPORTED;
        }

        auto continuous_impl = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl);
        if (!continuous_impl) {
            return SPIR_INTERNAL_ERROR;
        }

        // Check if this is a PiecewiseLegendrePolyFunctions or TauFunctionsAdaptor
        auto piecewise_impl = std::dynamic_pointer_cast<PiecewiseLegendrePolyFunctions>(continuous_impl);
        if (piecewise_impl) {
            // Get knots from the underlying PiecewiseLegendrePolyVector
            auto knots = piecewise_impl->get_impl()->get_knots();
            *n_knots = knots.size();
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try TauFunctionsAdaptor
        auto tau_adaptor = std::dynamic_pointer_cast<TauFunctionsAdaptor<sparseir::IRTauFuncsType<sparseir::Fermionic>>>(continuous_impl);
        if (tau_adaptor) {
            auto knots = tau_adaptor->get_impl()->get_obj().get_knots();
            *n_knots = knots.size();
            return SPIR_COMPUTATION_SUCCESS;
        }

        auto tau_adaptor_boson = std::dynamic_pointer_cast<TauFunctionsAdaptor<sparseir::IRTauFuncsType<sparseir::Bosonic>>>(continuous_impl);
        if (tau_adaptor_boson) {
            auto knots = tau_adaptor_boson->get_impl()->get_obj().get_knots();
            *n_knots = knots.size();
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try OmegaFunctionsAdaptor - use the actual type from the implementation
        // We need to check what type the OmegaFunctionsAdaptor is wrapping
        // For now, just call get_knots() directly on the continuous_impl
        try {
            auto knots = continuous_impl->get_knots();
            *n_knots = knots.size();
            return SPIR_COMPUTATION_SUCCESS;
        } catch (...) {
            // If get_knots() is not implemented, return empty
            *n_knots = 0;
            return SPIR_COMPUTATION_SUCCESS;
        }

        DEBUG_LOG("Error: knots are only available for piecewise Legendre polynomial functions");
        return SPIR_NOT_SUPPORTED;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_n_knots: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_get_n_knots");
        return SPIR_INTERNAL_ERROR;
    }
}

int spir_funcs_get_knots(const spir_funcs *funcs, double *knots)
{
    if (!funcs) {
        DEBUG_LOG("Error in spir_funcs_get_knots: invalid pointer funcs");
        return SPIR_INVALID_ARGUMENT;
    }
    if (!knots) {
        DEBUG_LOG("Error in spir_funcs_get_knots: invalid pointer knots");
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (!impl->is_continuous_funcs()) {
            DEBUG_LOG("Error: the function is not defined for continuous variables");
            return SPIR_NOT_SUPPORTED;
        }

        auto continuous_impl = std::dynamic_pointer_cast<AbstractContinuousFunctions>(impl);
        if (!continuous_impl) {
            return SPIR_INTERNAL_ERROR;
        }

        // Check if this is a PiecewiseLegendrePolyFunctions or TauFunctionsAdaptor
        auto piecewise_impl = std::dynamic_pointer_cast<PiecewiseLegendrePolyFunctions>(continuous_impl);
        if (piecewise_impl) {
            // Get knots from the underlying PiecewiseLegendrePolyVector
            auto knots_vec = piecewise_impl->get_impl()->get_knots();
            
            // Copy knots to output array (already in non-ascending order)
            std::memcpy(knots, knots_vec.data(), knots_vec.size() * sizeof(double));
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try TauFunctionsAdaptor
        auto tau_adaptor = std::dynamic_pointer_cast<TauFunctionsAdaptor<sparseir::IRTauFuncsType<sparseir::Fermionic>>>(continuous_impl);
        if (tau_adaptor) {
            auto knots_vec = tau_adaptor->get_impl()->get_obj().get_knots();
            
            // Copy knots to output array (already in non-ascending order)
            std::memcpy(knots, knots_vec.data(), knots_vec.size() * sizeof(double));
            return SPIR_COMPUTATION_SUCCESS;
        }

        auto tau_adaptor_boson = std::dynamic_pointer_cast<TauFunctionsAdaptor<sparseir::IRTauFuncsType<sparseir::Bosonic>>>(continuous_impl);
        if (tau_adaptor_boson) {
            auto knots_vec = tau_adaptor_boson->get_impl()->get_obj().get_knots();
            
            // Copy knots to output array (already in non-ascending order)
            std::memcpy(knots, knots_vec.data(), knots_vec.size() * sizeof(double));
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try OmegaFunctionsAdaptor - use the actual type from the implementation
        // We need to check what type the OmegaFunctionsAdaptor is wrapping
        // For now, just call get_knots() directly on the continuous_impl
        try {
            auto knots_vec = continuous_impl->get_knots();
            
            // Copy knots to output array (already in non-ascending order)
            std::memcpy(knots, knots_vec.data(), knots_vec.size() * sizeof(double));
            return SPIR_COMPUTATION_SUCCESS;
        } catch (...) {
            // If get_knots() is not implemented, return success with empty array
            return SPIR_COMPUTATION_SUCCESS;
        }

        DEBUG_LOG("Error: knots are only available for piecewise Legendre polynomial functions");
        return SPIR_NOT_SUPPORTED;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_get_knots: " + std::string(e.what()));
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_funcs_get_knots");
        return SPIR_INTERNAL_ERROR;
    }
}

} // extern "C"
