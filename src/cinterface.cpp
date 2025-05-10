#include "sparseir/sparseir.h"
#include "sparseir/sparseir.hpp"
#include "sparseir/utils.hpp"
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <iostream>

// Debug macro
//#ifdef DEBUG_CINTERFACE
#define DEBUG_LOG(msg) std::cerr << "[DEBUG] " << msg << std::endl
//#else
//#define DEBUG_LOG(msg)
//#endif

#include "cinterface_impl/helper_types.hpp"
#include "cinterface_impl/opaque_types.hpp"
#include "cinterface_impl/helper_funcs.hpp"

// Implementation of the C API
extern "C" {

int32_t spir_check_kernel_ptr(const spir_kernel* kernel)
{
    std::cout << "kernel ptr = " << kernel << std::endl;
    std::cout << "kernel->ptr.get() = " << kernel->ptr.get() << std::endl;
    std::cout << "kernel->ptr.use_count() = " << kernel->ptr.use_count() << std::endl;
    return SPIR_COMPUTATION_SUCCESS;
}

int32_t spir_logistic_kernel_new(spir_kernel** kernel, double lambda)
{
    DEBUG_LOG("Creating LogisticKernel with lambda=" << lambda);
    try {
        std::cout << "spir_logistic_kernel_new: orignal ptr = " << *kernel << std::endl;
        auto kernel_ptr = std::make_shared<sparseir::LogisticKernel>(lambda);
        std::cout << "kernel_ptr.use_count()=" << kernel_ptr.use_count() << std::endl;
        std::shared_ptr<sparseir::AbstractKernel> abstract_kernel = std::static_pointer_cast<sparseir::AbstractKernel>(kernel_ptr);
        
        // Debug output before creating the C interface object
        std::cout << "C++ kernel_ptr address: " << kernel_ptr.get() 
                  << " use_count: " << kernel_ptr.use_count() << std::endl;
        
        // Check if dynamic_cast works at this point
        auto check_logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(abstract_kernel);
        std::cout << "dynamic_cast to LogisticKernel before C interface: " 
                  << (check_logistic ? "succeeded" : "failed") << std::endl;
        
        *kernel = create_kernel(abstract_kernel);
        DEBUG_LOG("Created LogisticKernel at " << *kernel << ", raw ptr=" << (*kernel)->ptr.get());
        std::cout << "kernel_ptr.use_count() after create_kernel: " << kernel_ptr.use_count() << std::endl;
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_logistic_kernel_new: " << e.what());
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_logistic_kernel_new");
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_regularized_bose_kernel_new(spir_kernel** kernel, double lambda)
{
    DEBUG_LOG("Creating RegularizedBoseKernel with lambda=" << lambda);
    try {
        auto kernel_ptr = std::make_shared<sparseir::RegularizedBoseKernel>(lambda);
        auto abstract_kernel = std::static_pointer_cast<sparseir::AbstractKernel>(kernel_ptr);
        *kernel = create_kernel(abstract_kernel);
        DEBUG_LOG("Created RegularizedBoseKernel at " << *kernel << ", ptr=" << (*kernel)->ptr.get());
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_regularized_bose_kernel_new: " << e.what());
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_regularized_bose_kernel_new");
        return SPIR_INTERNAL_ERROR;
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

    std::cout << "spir_kernel_domain: impl=" << impl << std::endl;
    std::cout << "impl.get()=" << impl.get() << std::endl;
    std::cout << "impl.use_count()=" << impl.use_count() << std::endl;

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

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_kernel_domain: " << e.what());
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_kernel_domain");
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_sve_result_new(spir_sve_result** sve, const spir_kernel *k, double epsilon)
{
    try {
        // Debug info for the C kernel object received from Fortran
        std::cout << "Received kernel from Fortran: " << k << std::endl;
        if (k) {
            std::cout << "Kernel->ptr: " << k->ptr.get() << " use_count: " << k->ptr.use_count() << std::endl;
        }
        
        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl) {
            std::cout << "get_impl_kernel failed, returned nullptr" << std::endl;
            return SPIR_GET_IMPL_FAILED;
        }
        
        std::cout << "impl=" << impl << " use_count: " << impl.use_count() << std::endl;
        std::cout << "impl.get()=" << impl.get() << std::endl;
        
        // Check dynamic_cast results without using typeid
        auto check_logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(impl);
        auto check_bose = std::dynamic_pointer_cast<sparseir::RegularizedBoseKernel>(impl);
        
        std::cout << "dynamic_cast to LogisticKernel: " << (check_logistic ? "succeeded" : "failed") << std::endl;
        std::cout << "dynamic_cast to RegularizedBoseKernel: " << (check_bose ? "succeeded" : "failed") << std::endl;
        
        // Original typeid check - this might fail
        try {
            if (impl.get()) {
                std::cout << "impl.get() type=" << typeid(*impl.get()).name() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "Exception when using typeid: " << e.what() << std::endl;
        } catch (...) {
            std::cout << "Unknown exception when using typeid" << std::endl;
        }
        
        std::shared_ptr<sparseir::SVEResult> sve_result;
        
        if (auto logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(impl)) {
            sve_result = std::make_shared<sparseir::SVEResult>(sparseir::compute_sve(*logistic, epsilon));
        } else if (auto bose = std::dynamic_pointer_cast<sparseir::RegularizedBoseKernel>(impl)) {
            sve_result = std::make_shared<sparseir::SVEResult>(sparseir::compute_sve(*bose, epsilon));
        } else {
            return SPIR_INTERNAL_ERROR;
        }

        *sve = create_sve_result(sve_result);
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_sve_result_new: " << e.what());
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_sve_result_new");
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_finite_temp_basis_new(spir_finite_temp_basis** b, spir_statistics_type statistics, double beta,
                           double omega_max, double epsilon)
{
    try {
        auto kernel = sparseir::LogisticKernel(beta * omega_max);
        auto sve_result = sparseir::compute_sve(kernel, epsilon);
        if (statistics == SPIR_STATISTICS_FERMIONIC) {
            using FiniteTempBasisType =
                sparseir::FiniteTempBasis<sparseir::Fermionic>;
            auto impl =
                std::make_shared<FiniteTempBasisType>(beta, omega_max, epsilon, kernel, sve_result);
            *b = create_finite_temp_basis(
                std::make_shared<_FiniteTempBasis<sparseir::Fermionic>>(impl));
        } else {
            using FiniteTempBasisType =
                sparseir::FiniteTempBasis<sparseir::Bosonic>;
            auto impl =
                std::make_shared<FiniteTempBasisType>(beta, omega_max, epsilon, kernel, sve_result);
            *b = create_finite_temp_basis(
                std::make_shared<_FiniteTempBasis<sparseir::Bosonic>>(impl));
        }
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_finite_temp_basis_new: " << e.what());
        return SPIR_INTERNAL_ERROR;
    } catch (...) {
        DEBUG_LOG("Unknown exception in spir_finite_temp_basis_new");
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_finite_temp_basis_new_with_sve(
    spir_finite_temp_basis** b, spir_statistics_type statistics, double beta, double omega_max,
    const spir_kernel *k, const spir_sve_result *sve)
{
    try {
        // Get the kernel implementation
        std::shared_ptr<sparseir::AbstractKernel> impl = get_impl_kernel(k);
        if (!impl)
            return SPIR_GET_IMPL_FAILED;

        // Get the SVE result implementation
        auto sve_impl = get_impl_sve_result(sve);
        if (!sve_impl)
            return SPIR_GET_IMPL_FAILED;

        // switch on kernel type
        if (auto logistic = std::dynamic_pointer_cast<sparseir::LogisticKernel>(impl)) {
            *b = _spir_finite_temp_basis_new_with_sve(statistics, beta, omega_max, *logistic, sve);
        } else if (auto bose = std::dynamic_pointer_cast<sparseir::RegularizedBoseKernel>(impl)) {
            *b = _spir_finite_temp_basis_new_with_sve(statistics, beta, omega_max, *bose, sve);
        } else {
            DEBUG_LOG("Unknown kernel type");
            return SPIR_INVALID_ARGUMENT;
        }

        if (!*b) {
            DEBUG_LOG("Failed to create finite temperature basis");
            return SPIR_INTERNAL_ERROR;
        }

        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception& e) {
        DEBUG_LOG("Exception in spir_finite_temp_basis_new_with_sve: " << e.what());
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_tau_sampling_new(spir_sampling** s, const spir_finite_temp_basis *b)
{
    spir_statistics_type stat;
    int32_t status = spir_finite_temp_basis_get_statistics(b, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *s = _spir_sampling_new<sparseir::Fermionic,
                                  sparseir::TauSampling<sparseir::Fermionic>>(
            b);
        return SPIR_COMPUTATION_SUCCESS;
    } else {
        *s = _spir_sampling_new<sparseir::Bosonic,
                                  sparseir::TauSampling<sparseir::Bosonic>>(b);
        return SPIR_COMPUTATION_SUCCESS;
    }
}

int32_t spir_matsubara_sampling_new(spir_sampling** s, const spir_finite_temp_basis *b, bool positive_only)
{
    spir_statistics_type stat;
    int32_t status = spir_finite_temp_basis_get_statistics(b, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *s = _spir_matsubara_sampling_new<
            sparseir::Fermionic,
            sparseir::MatsubaraSampling<sparseir::Fermionic>>(b, positive_only);
        return SPIR_COMPUTATION_SUCCESS;
    } else {
        *s = _spir_matsubara_sampling_new<
            sparseir::Bosonic, sparseir::MatsubaraSampling<sparseir::Bosonic>>(
            b, positive_only);
        return SPIR_COMPUTATION_SUCCESS;
    }
}

int32_t spir_matsubara_sampling_get_sampling_points(const spir_sampling *s, int32_t n_points, int32_t *points)
{
    auto impl = get_impl_sampling(s);
    spir_statistics_type stat;
    int32_t status = spir_sampling_get_statistics(s, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (stat == SPIR_STATISTICS_FERMIONIC) {
        auto smpl = std::dynamic_pointer_cast<sparseir::MatsubaraSampling<sparseir::Fermionic>>(impl);
        auto smpl_points = smpl->sampling_points();
        if (n_points != smpl_points.size())
            return SPIR_INVALID_ARGUMENT;

        std::vector<int> smpl_points_vec;
        for (int32_t i = 0; i < n_points; i++) {
            smpl_points_vec.push_back(smpl_points[i].get_n());
        }
        for (int32_t i = 0; i < n_points; i++) {
            points[i] = smpl_points_vec[i];
        }
    } else {
        auto smpl = std::dynamic_pointer_cast<sparseir::MatsubaraSampling<sparseir::Bosonic>>(impl);
        auto smpl_points = smpl->sampling_points();
        if (n_points != smpl_points.size())
            return SPIR_INVALID_ARGUMENT;

        std::vector<int> smpl_points_vec;
        for (int32_t i = 0; i < n_points; i++) {
            smpl_points_vec.push_back(smpl_points[i].get_n());
        }
        for (int32_t i = 0; i < n_points; i++) {
            points[i] = smpl_points_vec[i];
        }
    }

    return SPIR_COMPUTATION_SUCCESS;
}

int32_t spir_matsubara_sampling_get_num_points(const spir_sampling *s, int32_t *n_points)
{
    auto impl = get_impl_sampling(s);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }
    try {
        *n_points = impl->n_sampling_points();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_matsubara_sampling_dlr_new(spir_sampling** s, const spir_dlr *dlr, int32_t n_smpl_points, const int32_t *smpl_points, bool positive_only)
{
    spir_statistics_type stat;
    int32_t status = spir_dlr_get_statistics(dlr, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return SPIR_GET_IMPL_FAILED;
    }
    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *s = _spir_matsubara_sampling_dlr_new<
            sparseir::Fermionic,
            sparseir::MatsubaraSampling<sparseir::Fermionic>>(dlr, n_smpl_points, smpl_points, positive_only);
        return SPIR_COMPUTATION_SUCCESS;
    } else {
        *s = _spir_matsubara_sampling_dlr_new<
            sparseir::Bosonic, sparseir::MatsubaraSampling<sparseir::Bosonic>>(
            dlr, n_smpl_points, smpl_points, positive_only);
        return SPIR_COMPUTATION_SUCCESS;
    }
}


int32_t spir_dlr_new(spir_dlr** dlr, const spir_finite_temp_basis *b)
{
    spir_statistics_type stat;
    int32_t status = spir_finite_temp_basis_get_statistics(b, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *dlr = _spir_dlr_new<sparseir::Fermionic>(b);
        return SPIR_COMPUTATION_SUCCESS;
    } else {
        *dlr = _spir_dlr_new<sparseir::Bosonic>(b);
        return SPIR_COMPUTATION_SUCCESS;
    }
}

int32_t spir_dlr_new_with_poles(spir_dlr** dlr, const spir_finite_temp_basis *b,
                                  const int npoles, const double *poles)
{
    auto impl = get_impl_finite_temp_basis(b);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    spir_statistics_type stat;
    int32_t status = spir_finite_temp_basis_get_statistics(b, &stat);
    if (status != SPIR_COMPUTATION_SUCCESS) {
        return SPIR_GET_IMPL_FAILED;
    }

    if (stat == SPIR_STATISTICS_FERMIONIC) {
        *dlr = _spir_dlr_new_with_poles<sparseir::Fermionic>(b, npoles, poles);
        return SPIR_COMPUTATION_SUCCESS;
    } else {
        *dlr = _spir_dlr_new_with_poles<sparseir::Bosonic>(b, npoles, poles);
        return SPIR_COMPUTATION_SUCCESS;
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

int32_t spir_dlr_to_IR_dd(const spir_dlr *dlr, spir_order_type order, int32_t ndim,
                       const int32_t *input_dims, int32_t target_dim,
                       const double *input, double *out)
{
    auto impl = get_impl_dlr(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr_to_IR<sparseir::Fermionic, double>(dlr, order, ndim, input_dims, target_dim,
                                                   input, out);
    } else {
        return spir_dlr_to_IR<sparseir::Bosonic, double>(dlr, order, ndim, input_dims, target_dim,
                                                 input, out);
    }
}

int32_t spir_dlr_to_IR_zz(const spir_dlr *dlr, spir_order_type order, int32_t ndim,
                       const int32_t *input_dims, int32_t target_dim,
                       const c_complex *input, c_complex *out)
{
    auto impl = get_impl_dlr(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr_to_IR<sparseir::Fermionic, std::complex<double>>(dlr, order, ndim, input_dims, target_dim,
                                                   cpp_input, cpp_out);
    } else {
        return spir_dlr_to_IR<sparseir::Bosonic, std::complex<double>>(dlr, order, ndim, input_dims, target_dim,
                                                 cpp_input, cpp_out);
    }
}

int32_t spir_dlr_from_IR_dd(const spir_dlr *dlr, spir_order_type order,
                         int32_t ndim, const int32_t *input_dims, int32_t target_dim,
                         const double *input, double *out)
{
    auto impl = get_impl_dlr(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;
    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr_from_IR<sparseir::Fermionic, double>(dlr, order, ndim,
                                                     input_dims, target_dim, input, out);
    } else {
        return spir_dlr_from_IR<sparseir::Bosonic, double>(dlr, order, ndim, input_dims,
                                                   target_dim, input, out);
    }
}

int32_t spir_dlr_from_IR_zz(const spir_dlr *dlr, spir_order_type order,
                         int32_t ndim, const int32_t *input_dims, int32_t target_dim,
                         const c_complex *input, c_complex *out)
{
    auto impl = get_impl_dlr(dlr);
    if  (!impl)
        return SPIR_GET_IMPL_FAILED;

    std::complex<double> *cpp_input = (std::complex<double> *)(input);
    std::complex<double> *cpp_out = (std::complex<double> *)(out);

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return spir_dlr_from_IR<sparseir::Fermionic, std::complex<double>>(dlr, order, ndim,
                                                     input_dims, target_dim, cpp_input, cpp_out);
    } else {
        return spir_dlr_from_IR<sparseir::Bosonic, std::complex<double>>(dlr, order, ndim, input_dims,
                                                   target_dim, cpp_input, cpp_out);
    }
}

int32_t spir_dlr_get_num_poles(const spir_dlr *dlr, int32_t *num_poles)
{
    if (!dlr || !num_poles) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_dlr(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        *num_poles = impl->size();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_dlr_get_poles(const spir_dlr *dlr, double *poles)
{
    if (!dlr || !poles) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_dlr(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        std::vector<double> poles_vec = impl->get_poles();
        std::memcpy(poles, poles_vec.data(), poles_vec.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_dlr_get_statistics(const spir_dlr *dlr,
                                spir_statistics_type *statistics)
{
    if (!dlr || !statistics) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_dlr(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        *statistics = impl->get_statistics();
        return SPIR_COMPUTATION_SUCCESS;
    } catch (...) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_dlr_get_u(const spir_dlr *dlr, spir_funcs **u)
{
    if (!dlr || !u) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_dlr(dlr);
    if (!impl)
        return SPIR_GET_IMPL_FAILED;

    if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
        return _spir_dlr_get_u<sparseir::Fermionic>(dlr, u);
    } else {
        return _spir_dlr_get_u<sparseir::Bosonic>(dlr, u);
    }
}

int32_t spir_dlr_get_uhat(const spir_dlr* dlr, spir_matsubara_funcs** uhat)
{
    if (!dlr || !uhat) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_dlr(dlr);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            return _spir_dlr_get_uhat<sparseir::Fermionic>(dlr, uhat);
        } else {
            return _spir_dlr_get_uhat<sparseir::Bosonic>(dlr, uhat);
        }
    } catch (const std::exception& e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_finite_temp_basis_get_v(const spir_finite_temp_basis *b,
                                     spir_funcs **v)
{
    if (!b || !v) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_finite_temp_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            return _spir_finite_temp_basis_get_v<sparseir::Fermionic>(b, v);
        } else {
            return _spir_finite_temp_basis_get_v<sparseir::Bosonic>(b, v);
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

int32_t spir_finite_temp_basis_get_uhat(const spir_finite_temp_basis *b,
                                        spir_matsubara_funcs **uhat)
{
    if (!b || !uhat) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_finite_temp_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            return _spir_finite_temp_basis_get_uhat<sparseir::Fermionic>(b,
                                                                         uhat);
        } else {
            return _spir_finite_temp_basis_get_uhat<sparseir::Bosonic>(b, uhat);
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
    }
}

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

int32_t spir_finite_temp_basis_get_size(const spir_finite_temp_basis *b,
                                        int32_t *size)
{
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

int32_t spir_finite_temp_basis_get_statistics(const spir_finite_temp_basis *b,
                                              spir_statistics_type *statistics)
{
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

int32_t spir_sampling_get_statistics(const spir_sampling *s,
                                     spir_statistics_type *statistics)
{
    auto impl = get_impl_sampling(s);
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

int32_t spir_evaluate_funcs(const spir_funcs *funcs, double x, double *out)
{
    if (!funcs || !out) {
        return SPIR_INVALID_ARGUMENT;
    }

    try {
        Eigen::VectorXd result = funcs->ptr->operator()(x);
        std::memcpy(out, result.data(), result.size() * sizeof(double));
        return SPIR_COMPUTATION_SUCCESS;
    } catch (const std::exception &e) {
        DEBUG_LOG("Exception in spir_evaluate_funcs: " << e.what());
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_evaluate_matsubara_funcs(const spir_matsubara_funcs *uiw,
                                          spir_order_type order,
                                          int32_t num_freqs,
                                          int32_t *matsubara_freq_indices,
                                          c_complex *out)
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
        Eigen::VectorXi freq_indices =
            Eigen::Map<Eigen::VectorXi>(matsubara_freq_indices, num_freqs);

        // Get the func size
        int func_size = uiw->ptr->size();

        // Evaluate functions at all frequencies
        // The operator() returns a matrix of shape (nfuncs, nfreqs)
        Eigen::MatrixXcd out_matrix = uiw->ptr->operator()(freq_indices);

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
        DEBUG_LOG("Exception in spir_evaluate_matsubara_funcs: " << e.what());
        return SPIR_INTERNAL_ERROR;
    }
}

int32_t spir_finite_temp_basis_get_u(const spir_finite_temp_basis *b,
                                     spir_funcs **u)
{
    if (!b || !u) {
        return SPIR_INVALID_ARGUMENT;
    }

    auto impl = get_impl_finite_temp_basis(b);
    if (!impl) {
        return SPIR_GET_IMPL_FAILED;
    }

    try {
        if (impl->get_statistics() == SPIR_STATISTICS_FERMIONIC) {
            return _spir_finite_temp_basis_get_u<sparseir::Fermionic>(b, u);
        } else {
            return _spir_finite_temp_basis_get_u<sparseir::Bosonic>(b, u);
        }
    } catch (const std::exception &e) {
        return SPIR_GET_IMPL_FAILED;
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

int32_t spir_matsubara_funcs_get_size(const spir_matsubara_funcs* funcs, int32_t* size) {
    return _spir_matsubara_funcs_get_size<double>(funcs, size);
}

} // extern "C"
