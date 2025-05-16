#include "sparseir/sparseir.h"
#include "sparseir/sparseir.hpp"
#include "sparseir/utils.hpp"
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <iostream>

// Debug macro
#ifdef DEBUG_CINTERFACE
#define DEBUG_LOG(msg) std::cerr << "[DEBUG] " << msg << std::endl
#else
#define DEBUG_LOG(msg)
#endif

#include "cinterface_impl/helper_types.hpp"
#include "cinterface_impl/opaque_types.hpp"
#include "cinterface_impl/helper_funcs.hpp"

// Implementation of the C API
extern "C" {

spir_kernel* spir_logistic_kernel_new(double lambda, int32_t* status)
{
    try {
        auto kernel_ptr = std::make_shared<sparseir::LogisticKernel>(lambda);
        std::shared_ptr<sparseir::AbstractKernel> abstract_kernel = std::static_pointer_cast<sparseir::AbstractKernel>(kernel_ptr);
        
        // Check if dynamic_cast works at this point
        auto check_logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(abstract_kernel);
        
        *status = SPIR_COMPUTATION_SUCCESS;
        return create_kernel(abstract_kernel);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_logistic_kernel_new: " << e.what());
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_logistic_kernel_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_kernel* spir_regularized_bose_kernel_new(double lambda, int32_t* status)
{
    DEBUG_LOG("Creating RegularizedBoseKernel with lambda=" << lambda);
    try {
        auto kernel_ptr = std::make_shared<sparseir::RegularizedBoseKernel>(lambda);
        auto abstract_kernel = std::static_pointer_cast<sparseir::AbstractKernel>(kernel_ptr);
        *status = SPIR_COMPUTATION_SUCCESS;
        return create_kernel(abstract_kernel);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_regularized_bose_kernel_new: " << e.what());
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_regularized_bose_kernel_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

int32_t spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                           double *ymin, double *ymax)
{
    DEBUG_LOG("spir_kernel_domain called with kernel=" << k);
    auto impl = get_impl_kernel(k);
    if (!impl) {
        DEBUG_LOG("Failed to get kernel implementation");
        return SPIR_GET_IMPL_FAILED;
    }

    std::cout << "xmin=" << xmin << std::endl;

    try {
        DEBUG_LOG("Getting xrange and yrange");
        auto xrange = impl->xrange();
        auto yrange = impl->yrange();

        DEBUG_LOG("Setting output values: xrange=("
                  << xrange.first << ", " << xrange.second << "), yrange=("
                  << yrange.first << ", " << yrange.second << ")");
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
        DEBUG_LOG("Exception in spir_kernel_domain: " << e.what());
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_kernel_domain");
        return SPIR_INTERNAL_ERROR;
    }
}

spir_sve_result* spir_sve_result_new(const spir_kernel *k, double epsilon, int32_t* status)
{
    try {
        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }
        
        std::shared_ptr<sparseir::SVEResult> sve_result;
        
        if (auto logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(impl)) {
            sve_result = std::make_shared<sparseir::SVEResult>(sparseir::compute_sve(*logistic, epsilon));
        } else if (auto bose = std::dynamic_pointer_cast<sparseir::RegularizedBoseKernel>(impl)) {
            sve_result = std::make_shared<sparseir::SVEResult>(sparseir::compute_sve(*bose, epsilon));
        } else {
            *status = SPIR_INTERNAL_ERROR;
            return nullptr;
        }

        *status = SPIR_COMPUTATION_SUCCESS;
        return create_sve_result(sve_result);
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_new: " << e.what());
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_basis* spir_basis_new(spir_statistics_type statistics, double beta,
                           double omega_max, double epsilon, int32_t* status)
{
    try {
        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        auto sve_result = sparseir::compute_sve(kernel, epsilon);
        if (statistics == SPIR_STATISTICS_FERMIONIC) {
            using FiniteTempBasisType =
                sparseir::FiniteTempBasis<sparseir::Fermionic>;
            auto impl =
                std::make_shared<FiniteTempBasisType>(beta, omega_max, epsilon, kernel, sve_result);
            *status = SPIR_COMPUTATION_SUCCESS;
            return create_basis(
                std::make_shared<_FiniteTempBasis<sparseir::Fermionic>>(impl));
        } else {
            using FiniteTempBasisType =
                sparseir::FiniteTempBasis<sparseir::Bosonic>;
            auto impl =
                std::make_shared<FiniteTempBasisType>(beta, omega_max, epsilon, kernel, sve_result);
            *status = SPIR_COMPUTATION_SUCCESS;
            return create_basis(
                std::make_shared<_FiniteTempBasis<sparseir::Bosonic>>(impl));
        }
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_basis_new: " << e.what());
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_basis_new");
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_basis* spir_basis_new_with_sve(
    spir_statistics_type statistics, double beta, double omega_max,
    const spir_kernel *k, const spir_sve_result *sve, int32_t* status)
{
    try {
        // Get the kernel implementation
        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }

        // Get the SVE result implementation
        auto sve_impl = get_impl_sve_result(sve);
        if (!sve_impl) {
            *status = SPIR_GET_IMPL_FAILED;
            return nullptr;
        }

        // switch on kernel type
        spir_basis* result = nullptr;
        if (auto logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(impl)) {
            result = _spir_basis_new_with_sve(statistics, beta, omega_max, *logistic, sve);
        } else if (auto bose = std::dynamic_pointer_cast<sparseir::RegularizedBoseKernel>(impl)) {
            result = _spir_basis_new_with_sve(statistics, beta, omega_max, *bose, sve);
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
        DEBUG_LOG("Exception in spir_basis_new_with_sve: " << e.what());
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}

spir_sampling* spir_tau_sampling_new(const spir_basis *b, int32_t* status)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
    if (!is_ir_basis(b)) {
        std::cerr << "Error: The basis is not an IR basis" << std::endl;
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    spir_statistics_type stat;
    int32_t stat_status = spir_basis_get_statistics(b, &stat);
    if (stat_status != SPIR_COMPUTATION_SUCCESS) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_sampling_new<sparseir::Fermionic,
                                  sparseir::TauSampling<sparseir::Fermionic>>(
            b);
    } else {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_sampling_new<sparseir::Bosonic,
                                  sparseir::TauSampling<sparseir::Bosonic>>(b);
    }
}

spir_sampling* spir_matsubara_sampling_new(const spir_basis *b, bool positive_only, int32_t* status)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
    if (!is_ir_basis(b)) {
        std::cerr << "Error: The basis is not an IR basis" << std::endl;
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    spir_statistics_type stat;
    int32_t stat_status = spir_basis_get_statistics(b, &stat);
    if (stat_status != SPIR_COMPUTATION_SUCCESS) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_matsubara_sampling_new<
            sparseir::Fermionic,
            sparseir::MatsubaraSampling<sparseir::Fermionic>>(b, positive_only);
    } else {
        *status = SPIR_COMPUTATION_SUCCESS;
        return _spir_matsubara_sampling_new<
            sparseir::Bosonic, sparseir::MatsubaraSampling<sparseir::Bosonic>>(
            b, positive_only);
    }
}


spir_basis* spir_dlr_new(const spir_basis *b, int32_t* status)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        *status = SPIR_GET_IMPL_FAILED;
        return nullptr;
    }
    if (!is_ir_basis(b)) {
        std::cerr << "Error: The basis is not an IR basis" << std::endl;
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    spir_statistics_type stat;
    int32_t status_basis = spir_basis_get_statistics(b, &stat);
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
                                  const int32_t npoles, const double *poles, int32_t* status)
{
    auto impl = get_impl_basis(b);
    if (!impl)
        return nullptr;

    if (!is_ir_basis(b)) {
        std::cerr << "Error: The basis is not an IR basis" << std::endl;
        *status = SPIR_INVALID_ARGUMENT;
        return nullptr;
    }

    spir_statistics_type stat;
    int32_t status_basis = spir_basis_get_statistics(b, &stat);
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

int32_t spir_sampling_evaluate_dd(const spir_sampling *s, spir_order_type order,
                                  int32_t ndim, const int32_t *input_dims,
                                  int32_t target_dim, const double *input,
                                  double *out)
{
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, out,
                         &sparseir::AbstractSampling::evaluate_inplace_dd);
}

int32_t spir_sampling_evaluate_dz(const spir_sampling *s, spir_order_type order,
                                  int32_t ndim, const int32_t *input_dims,
                                  int32_t target_dim, const double *input,
                                  c_complex *out)
{
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, cpp_out,
                         &sparseir::AbstractSampling::evaluate_inplace_dz);
}

int32_t spir_sampling_evaluate_zz(const spir_sampling *s, spir_order_type order,
                                  int32_t ndim, const int32_t *input_dims,
                                  int32_t target_dim, const c_complex *input,
                                  c_complex *out)
{
    // DANGER: MEMORY LAYOUT MAY NOT BE CONSISTENT BETWEEN C99 AND C++
    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return evaluate_impl(s, order, ndim, input_dims, target_dim, cpp_input,
                         cpp_out,
                         &sparseir::AbstractSampling::evaluate_inplace_zz);
}

int32_t spir_sampling_fit_dd(const spir_sampling *s, spir_order_type order,
                             int32_t ndim, const int32_t *input_dims,
                             int32_t target_dim, const double *input,
                             double *out)
{
    return fit_impl(s, order, ndim, input_dims, target_dim, input, out,
                    &sparseir::AbstractSampling::fit_inplace_dd);
}

int32_t spir_sampling_fit_zz(const spir_sampling *s, spir_order_type order,
                             int32_t ndim, const int32_t *input_dims,
                             int32_t target_dim, const c_complex *input,
                             c_complex *out)
{
    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return fit_impl(s, order, ndim, input_dims, target_dim, cpp_input, cpp_out,
                    &sparseir::AbstractSampling::fit_inplace_zz);
}

int32_t spir_dlr_to_ir_dd(const spir_basis *dlr, spir_order_type order, int32_t ndim,
                       const int32_t *input_dims, int32_t target_dim,
                       const double *input, double *out)
{
    auto impl = get_impl_basis(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;
    
    if (!is_dlr_basis(dlr)) {
        std::cerr << "Error: The basis is not a DLR basis" << std::endl;
        return SPIR_INVALID_ARGUMENT;
    }

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr_to_ir<sparseir::Fermionic, double>(dlr, order, ndim, input_dims, target_dim,
                                                   input, out);
    } else {
        return spir_dlr_to_ir<sparseir::Bosonic, double>(dlr, order, ndim, input_dims, target_dim,
                                                 input, out);
    }
}

int32_t spir_dlr_to_ir_zz(const spir_basis *dlr, spir_order_type order, int32_t ndim,
                       const int32_t *input_dims, int32_t target_dim,
                       const c_complex *input, c_complex *out)
{
    auto impl = get_impl_basis(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    if (!is_dlr_basis(dlr)) {
        std::cerr << "Error: The basis is not a DLR basis" << std::endl;
        return SPIR_INVALID_ARGUMENT;
    }

    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr_to_ir<sparseir::Fermionic, std::complex<double>>(dlr, order, ndim, input_dims, target_dim,
                                                   cpp_input, cpp_out);
    } else {
        return spir_dlr_to_ir<sparseir::Bosonic, std::complex<double>>(dlr, order, ndim, input_dims, target_dim,
                                                 cpp_input, cpp_out);
    }
}

int32_t spir_ir_to_dlr_dd(const spir_basis *dlr, spir_order_type order,
                         int32_t ndim, const int32_t *input_dims, int32_t target_dim,
                         const double *input, double *out)
{
    auto impl = get_impl_basis(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;
    
    if (!is_dlr_basis(dlr)) {
        std::cerr << "Error: The basis is not a DLR basis" << std::endl;
        return SPIR_INVALID_ARGUMENT;
    }

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_ir_to_dlr<sparseir::Fermionic, double>(dlr, order, ndim,
                                                     input_dims, target_dim, input, out);
    } else {
        return spir_ir_to_dlr<sparseir::Bosonic, double>(dlr, order, ndim, input_dims,
                                                   target_dim, input, out);
    }
}

int32_t spir_ir_to_dlr_zz(const spir_basis *dlr, spir_order_type order,
                         int32_t ndim, const int32_t *input_dims, int32_t target_dim,
                         const c_complex *input, c_complex *out)
{
    auto impl = get_impl_basis(dlr);
    if  (!impl)
        return SPIR_GET_IMPL_FAILED;
    
    if (!is_dlr_basis(dlr)) {
        std::cerr << "Error: The basis is not a DLR basis" << std::endl;
        return SPIR_INVALID_ARGUMENT;
    }

    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_ir_to_dlr<sparseir::Fermionic, std::complex<double>>(dlr, order, ndim,
                                                     input_dims, target_dim, cpp_input, cpp_out);
    } else {
        return spir_ir_to_dlr<sparseir::Bosonic, std::complex<double>>(dlr, order, ndim, input_dims,
                                                   target_dim, cpp_input, cpp_out);
    }
}

int32_t spir_dlr_get_num_poles(const spir_basis *dlr, int32_t *num_poles)
{
    if (!dlr || !num_poles) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_dlr_basis(dlr)) {
        std::cerr << "Error: The basis is not a DLR basis" << std::endl;
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        *num_poles = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_dlr_get_poles(const spir_basis *dlr, double *poles)
{
    if (!dlr || !poles) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_basis(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (!is_dlr_basis(dlr)) {
        std::cerr << "Error: The basis is not a DLR basis" << std::endl;
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

spir_funcs* spir_basis_get_u(const spir_basis *b, int32_t *status)
{
    if (!b || !status) {
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
            int32_t ret = _spir_basis_get_u<sparseir::Fermionic>(b, &u);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return u;
        } else {
            spir_funcs *u = nullptr;
            int32_t ret = _spir_basis_get_u<sparseir::Bosonic>(b, &u);
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

spir_funcs* spir_basis_get_v(const spir_basis *b, int32_t *status)
{
    if (!b || !status) {
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
            int32_t ret = _spir_get_v<sparseir::Fermionic>(b, &v);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return v;
        } else {
            spir_funcs *v = nullptr;
            int32_t ret = _spir_get_v<sparseir::Bosonic>(b, &v);
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

spir_funcs* spir_basis_get_uhat(const spir_basis *b, int32_t *status)
{
    if (!b || !status) {
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
            int32_t ret = _spir_basis_get_uhat<sparseir::Fermionic>(b, &uhat);
            if (ret != SPIR_COMPUTATION_SUCCESS) {
                *status = ret;
                return nullptr;
            }
            *status = SPIR_COMPUTATION_SUCCESS;
            return uhat;
        } else {
            spir_funcs *uhat = nullptr;
            int32_t ret = _spir_basis_get_uhat<sparseir::Bosonic>(b, &uhat);
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
int32_t spir_sampling_get_num_points(const spir_sampling *s,
                                     int32_t *num_points)
{
    auto impl = get_impl_sampling(s);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (!num_points) {
        return SPIR_INVALID_ARGUMENT;
    }
    try {
        *num_points = impl->n_sampling_points();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_sampling_get_tau_points(const spir_sampling *s, double *points)
{
    auto impl = get_impl_sampling(s);
    if (!impl) {
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
int32_t spir_sampling_get_matsubara_points(const spir_sampling *s,
                                           int32_t *points)
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
                    return static_cast<int>(freq.get_n());
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
                    return static_cast<int>(freq.get_n());
                });
            return SPIR_COMPUTATION_SUCCESS;
        }

        return SPIR_NOT_SUPPORTED;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_basis_get_size(const spir_basis *b,
                                        int32_t *size)
{
    auto impl = get_impl_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (!size) {
        return SPIR_INVALID_ARGUMENT;
    }
    try {
        *size = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_basis_get_statistics(const spir_basis *b,
                                              spir_statistics_type *statistics)
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

int32_t spir_funcs_evaluate(const spir_funcs *funcs, double x, double *out)
{
    if (!out) {
        std::cerr << "Error: out is null" << std::endl;
        return SPIR_INVALID_ARGUMENT;
    }
    auto impl = get_impl_funcs(funcs);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (!impl->is_continuous_funcs()) {
        std::cerr << "Error: the function is not defined for continuous variables" << std::endl;
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        Eigen::VectorXd result = std::dynamic_pointer_cast<AbstractContinuousFunctions>(funcs->ptr)->operator()(x);
        std::memcpy(out, result.data(), result.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_funcs_evaluate: " << e.what());
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_funcs_evaluate_matsubara(const spir_funcs *uiw,
                                          spir_order_type order,
                                          int32_t num_freqs,
                                          int32_t *matsubara_freq_indices,
                                          c_complex *out)
{
    auto impl = get_impl_funcs(uiw);
    if (!impl) {
        DEBUG_LOG("Matsubara basis functions object is null or not assigned");
        return SPIR_GET_IMPL_FAILED;
    }
    if (impl->is_continuous_funcs()) {
        std::cerr << "Error: the function is not defined for Matsubara frequencies" << std::endl;
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
        Eigen::VectorXi freq_indices =
            Eigen::Map<Eigen::VectorXi>(matsubara_freq_indices, num_freqs);

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
        DEBUG_LOG("Exception in spir_funcs_evaluate_matsubara: " << e.what());
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_funcs_get_size(const spir_funcs *funcs, int32_t *size)
{
    if (funcs == nullptr || size == nullptr) {
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

} // extern "C"
