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

spir_kernel *spir_logistic_kernel_new(double lambda)
{
    DEBUG_LOG("Creating LogisticKernel with lambda=" << lambda);
    try {
        auto kernel = std::make_shared<sparseir::LogisticKernel>(lambda);
        auto abstract_kernel =
            std::shared_ptr<sparseir::AbstractKernel>(kernel);
        auto result = create_kernel(abstract_kernel);
        DEBUG_LOG("Created LogisticKernel at "
                  << result << ", ptr=" << result->ptr.get());
        return result;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_logistic_kernel_new: " << e.what());
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_logistic_kernel_new");
        return nullptr;
    }
}

spir_kernel *spir_regularized_bose_kernel_new(double lambda)
{
    DEBUG_LOG("Creating RegularizedBoseKernel with lambda=" << lambda);
    try {
        auto kernel = std::make_shared<sparseir::RegularizedBoseKernel>(lambda);
        auto abstract_kernel =
            std::shared_ptr<sparseir::AbstractKernel>(kernel);
        auto result = create_kernel(abstract_kernel);
        DEBUG_LOG("Created RegularizedBoseKernel at "
                  << result << ", ptr=" << result->ptr.get());
        return result;
    } catch (const std::exception &e) {
        DEBUG_LOG(
            "Exception in spir_regularized_bose_kernel_new: " << e.what());
        return nullptr;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_regularized_bose_kernel_new");
        return nullptr;
    }
}

int spir_kernel_domain(const spir_kernel *k, double *xmin, double *xmax,
                       double *ymin, double *ymax)
{
    DEBUG_LOG("spir_kernel_domain called with kernel=" << k);
    auto impl = get_impl_kernel(k);
    if (!impl) {
        DEBUG_LOG("Failed to get kernel implementation");
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        DEBUG_LOG("Getting xrange and yrange");
        auto xrange = impl->xrange();
        auto yrange = impl->yrange();

        DEBUG_LOG("Setting output values: xrange=("
                  << xrange.first << ", " << xrange.second << "), yrange=("
                  << yrange.first << ", " << yrange.second << ")");
        *xmin = xrange.first;
        *xmax = xrange.second;
        *ymin = yrange.first;
        *ymax = yrange.second;

        return 0;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_kernel_domain: " << e.what());
        return -1;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_kernel_domain");
        return -1;
    }
}


spir_sve_result *spir_sve_result_new(const spir_kernel *k, double epsilon)
{
    try {
        auto impl = get_impl_kernel(k);
        if (!impl)
            return nullptr;
        auto sve = sparseir::compute_sve(impl, epsilon);
        return create_sve_result(std::make_shared<sparseir::SVEResult>(sve));
    } catch (...) {
        return nullptr;
    }
}

spir_finite_temp_basis *
spir_finite_temp_basis_new(
    spir_statistics_type statistics, double beta, double omega_max,
                                     double epsilon)
{
    try {
        if (statistics == SPIR_STATISTICS_FERMIONIC) {  
            using FiniteTempBasisType = sparseir::FiniteTempBasis<sparseir::Fermionic>;
            auto impl = std::make_shared<FiniteTempBasisType>(beta, omega_max, epsilon);
            return create_finite_temp_basis(std::make_shared<_FiniteTempBasis<sparseir::Fermionic>>(impl));
        } else {
            using FiniteTempBasisType = sparseir::FiniteTempBasis<sparseir::Bosonic>;
            auto impl = std::make_shared<FiniteTempBasisType>(beta, omega_max, epsilon);
            return create_finite_temp_basis(std::make_shared<_FiniteTempBasis<sparseir::Bosonic>>(impl));
        }
    } catch (...) {
        return nullptr;
    }
}

spir_finite_temp_basis *
spir_finite_temp_basis_new_with_sve(spir_statistics_type statistics, double beta, double omega_max,
                                              const spir_kernel *k,
                                              const spir_sve_result *sve)
{
    try {
        auto sve_impl = get_impl_sve_result(sve);
        auto kernel_impl = get_impl_kernel(k);
        if (!sve_impl || !kernel_impl)
            return nullptr;
        if (statistics == SPIR_STATISTICS_FERMIONIC) {
            using FiniteTempBasisType = sparseir::FiniteTempBasis<sparseir::Fermionic>;
            auto impl = std::make_shared<FiniteTempBasisType>(beta, omega_max, kernel_impl, *sve_impl);
            return create_finite_temp_basis(std::make_shared<_FiniteTempBasis<sparseir::Fermionic>>(impl));
        } else {
            using FiniteTempBasisType = sparseir::FiniteTempBasis<sparseir::Bosonic>;
            auto impl = std::make_shared<FiniteTempBasisType>(beta, omega_max, kernel_impl, *sve_impl);
            return create_finite_temp_basis(std::make_shared<_FiniteTempBasis<sparseir::Bosonic>>(impl));
        }
    } catch (...) {
        return nullptr;
    }
}


spir_sampling *
spir_tau_sampling_new(const spir_finite_temp_basis *b)
{
    spir_statistics_type stat;
    int32_t status = spir_finite_temp_basis_get_statistics(b, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return nullptr;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        return _spir_sampling_new<sparseir::Fermionic, sparseir::TauSampling<sparseir::Fermionic>>(b);
    } else {
        return _spir_sampling_new<sparseir::Bosonic, sparseir::TauSampling<sparseir::Bosonic>>(b);
    }
}

spir_sampling *
spir_matsubara_sampling_new(const spir_finite_temp_basis *b)
{
    spir_statistics_type stat;
    int32_t status = spir_finite_temp_basis_get_statistics(b, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return nullptr;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        return _spir_sampling_new<sparseir::Fermionic, sparseir::MatsubaraSampling<sparseir::Fermionic>>(b);
    } else {
        return _spir_sampling_new<sparseir::Bosonic, sparseir::MatsubaraSampling<sparseir::Bosonic>>(b);
    }
}


spir_dlr *
spir_dlr_new(const spir_finite_temp_basis *b)
{
    spir_statistics_type stat;
    int32_t status = spir_finite_temp_basis_get_statistics(b, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return nullptr;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        return _spir_dlr_new<sparseir::Fermionic>(b);
    } else {
        return _spir_dlr_new<sparseir::Bosonic>(b);
    }
}

spir_dlr *
spir_dlr_new_with_poles(const spir_finite_temp_basis *b, const int npoles, const double *poles)
{
    auto impl = get_impl_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    spir_statistics_type stat;
    int32_t status = spir_finite_temp_basis_get_statistics(b, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return nullptr;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        return _spir_dlr_new_with_poles<sparseir::Fermionic>(b, npoles, poles);
    } else {
        return _spir_dlr_new_with_poles<sparseir::Bosonic>(b, npoles, poles);
    }
}


int32_t spir_sampling_evaluate_dd(const spir_sampling *s, spir_order_type order,
                              int32_t ndim, int32_t *input_dims,
                              int32_t target_dim, const double *input,
                              double *out)
{
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, out,
                         &sparseir::AbstractSampling::evaluate_inplace_dd);
}

int32_t spir_sampling_evaluate_dz(const spir_sampling *s, spir_order_type order,
                              int32_t ndim, int32_t *input_dims,
                              int32_t target_dim, const double *input,
                              c_complex *out)
{
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return evaluate_impl(s, order, ndim, input_dims, target_dim, input, cpp_out,
                         &sparseir::AbstractSampling::evaluate_inplace_dz);
}

int32_t spir_sampling_evaluate_zz(const spir_sampling *s, spir_order_type order,
                              int32_t ndim, int32_t *input_dims,
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
                         int32_t ndim, int32_t *input_dims, int32_t target_dim,
                         const double *input, double *out)
{
    return fit_impl(s, order, ndim, input_dims, target_dim, input, out,
                    &sparseir::AbstractSampling::fit_inplace_dd);
}

int32_t spir_sampling_fit_zz(const spir_sampling *s, spir_order_type order,
                         int32_t ndim, int32_t *input_dims, int32_t target_dim,
                         const c_complex *input, c_complex *out)
{
    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);
    return fit_impl(s, order, ndim, input_dims, target_dim, cpp_input, cpp_out,
                    &sparseir::AbstractSampling::fit_inplace_zz);
}

int32_t spir_dlr_fitmat_rows(const spir_dlr *dlr)
{
    auto impl = get_impl_dlr(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;
    return impl->fitmat_rows();
}

int32_t spir_dlr_fitmat_cols(const spir_dlr *dlr)
{
    auto impl = get_impl_dlr(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;
    return impl->fitmat_cols();
}


int32_t spir_dlr_to_IR(const spir_dlr *dlr, spir_order_type order,
                           int32_t ndim, int32_t *input_dims,
                           const double *input, double *out)
{
    auto impl = get_impl_dlr(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;
    
    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr_to_IR<sparseir::Fermionic>(dlr, order, ndim, input_dims, input, out);
    } else {
        return spir_dlr_to_IR<sparseir::Bosonic>(dlr, order, ndim, input_dims, input, out);
    }
}


int32_t spir_dlr_from_IR(const spir_dlr *dlr, spir_order_type order,
                             int32_t ndim, int32_t *input_dims,
                             const double *input, double *out)
{

    auto impl = get_impl_dlr(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;
    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr_from_IR<sparseir::Fermionic>(dlr, order, ndim, input_dims, input, out);
    } else {
        return spir_dlr_from_IR<sparseir::Bosonic>(dlr, order, ndim, input_dims, input, out);
    }
}


spir_funcs *spir_finite_temp_basis_get_u(const spir_finite_temp_basis *b)
{
    auto impl = get_impl_finite_temp_basis(b);
    if (!impl)
        return nullptr;
    return _create_funcs(impl->get_u());
}


spir_funcs *spir_finite_temp_basis_get_v(const spir_finite_temp_basis *b)
{
    auto impl = get_impl_finite_temp_basis(b);
    if (!impl)
        return nullptr;
    return _create_funcs(impl->get_v());
}

spir_matsubara_functions *spir_finite_temp_basis_get_uhat(const spir_finite_temp_basis *b)
{
    auto impl = get_impl_finite_temp_basis(b);
    if (!impl)
        return nullptr;

    return create_matsubara_functions(impl->get_uhat());
}


int32_t spir_sampling_get_num_points(const spir_sampling *s, int32_t *num_points) {
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

int32_t spir_sampling_get_tau_points(const spir_sampling *s, double *points) {
    auto impl = get_impl_sampling(s);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    try {
        // Try Fermionic case first
        auto fermionic_sampling = std::dynamic_pointer_cast<sparseir::TauSampling<sparseir::Fermionic>>(impl);
        if (fermionic_sampling) {
            auto tau_points = fermionic_sampling->sampling_points();
            std::copy(tau_points.begin(), tau_points.end(), points);
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try Bosonic case
        auto bosonic_sampling = std::dynamic_pointer_cast<sparseir::TauSampling<sparseir::Bosonic>>(impl);
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

int32_t spir_sampling_get_matsubara_points(const spir_sampling *s, int32_t *points) {
    auto impl = get_impl_sampling(s);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    try {
        // Try Fermionic case first
        auto fermionic_sampling = std::dynamic_pointer_cast<sparseir::MatsubaraSampling<sparseir::Fermionic>>(impl);
        if (fermionic_sampling) {
            auto matsubara_points = fermionic_sampling->sampling_points();
            std::transform(matsubara_points.begin(), matsubara_points.end(), points,
                [](const sparseir::MatsubaraFreq<sparseir::Fermionic> &freq) { return static_cast<int>(freq.get_n()); });
            return SPIR_COMPUTATION_SUCCESS;
        }

        // Try Bosonic case
        auto bosonic_sampling = std::dynamic_pointer_cast<sparseir::MatsubaraSampling<sparseir::Bosonic>>(impl);
        if (bosonic_sampling) {
            auto matsubara_points = bosonic_sampling->sampling_points();
            std::transform(matsubara_points.begin(), matsubara_points.end(), points,
                [](const sparseir::MatsubaraFreq<sparseir::Bosonic> &freq) { return static_cast<int>(freq.get_n()); });
            return SPIR_COMPUTATION_SUCCESS;
        }

        return SPIR_NOT_SUPPORTED;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_finite_temp_basis_get_size(const spir_finite_temp_basis *b, int32_t *size) {
    auto impl = get_impl_finite_temp_basis(b);
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

int32_t spir_finite_temp_basis_get_statistics(const spir_finite_temp_basis *b, spir_statistics_type *statistics) {
    auto impl = get_impl_finite_temp_basis(b);
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

int32_t spir_evaluate_funcs(const spir_funcs* u, double x, double* out) {
    if (!u || !out) {
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        Eigen::VectorXd result = u->ptr->operator()(x);
        std::memcpy(out, result.data(), result.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception& e) {
        DEBUG_LOG("Exception in spir_evaluate_funcs: " << e.what());
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_evaluate_matsubara_functions(
    const spir_matsubara_functions* uiw,
    spir_order_type order,
    int32_t num_freqs,
    int32_t* matsubara_freq_indices,
    c_complex* out)
{
    if (!uiw || !uiw->ptr) {
        DEBUG_LOG("Matsubara basis functions object is null or not assigned");
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
        Eigen::VectorXi freq_indices = Eigen::Map<Eigen::VectorXi>(matsubara_freq_indices, num_freqs);
        
        // Get the basis size
        int basis_size = uiw->ptr->size();
        
        Eigen::MatrixXcd out_matrix = uiw->ptr->operator()(freq_indices);
        
        for (int ifreq = 0; ifreq < num_freqs; ++ifreq) {
            for (int ibasis = 0; ibasis < basis_size; ++ibasis) {
                const auto& val = out_matrix(ibasis, ifreq);
                if (order == SPIR_ORDER_ROW_MAJOR) {
                    out[ifreq + ibasis * num_freqs] = {val.real(), val.imag()};
                } else {
                    out[ibasis + ifreq * basis_size] = {val.real(), val.imag()};
                }
            }
        }

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception& e) {
        DEBUG_LOG("Exception in spir_evaluate_matsubarabasis_functions: " << e.what());
        return SPIR_INTERNAL_ERROR;
    }
}

} // extern "C"
