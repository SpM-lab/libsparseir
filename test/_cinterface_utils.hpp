#pragma once

#include <memory>

#include <sparseir/sparseir.h> // C interface

inline spir_basis *_spir_basis_new(int32_t statistics, double beta,
                                  double omega_max, double epsilon,
                                  int32_t *status)
{
    int32_t status_;
    spir_kernel *kernel = nullptr;
    spir_sve_result *sve = nullptr;
    spir_basis *basis = nullptr;

    try {
        // Create a logistic kernel
        kernel = spir_logistic_kernel_new(beta * omega_max, &status_);
        if (status_ != SPIR_COMPUTATION_SUCCESS || kernel == nullptr) {
            *status = status_;
            return nullptr;
        }

        // Create a pre-computed SVE result
        sve = spir_sve_result_new(kernel, epsilon, &status_);
        if (status_ != SPIR_COMPUTATION_SUCCESS || sve == nullptr) {
            *status = status_;
            spir_kernel_release(kernel);
            return nullptr;
        }

        // Create a fermionic finite temperature basis with pre-computed SVE result
        basis = spir_basis_new(
            statistics, beta, omega_max, kernel, sve, &status_);
        if (status_ != SPIR_COMPUTATION_SUCCESS || basis == nullptr) {
            *status = status_;
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
            return nullptr;
        }

        // Success case - clean up intermediate objects
        spir_sve_result_release(sve);
        spir_kernel_release(kernel);
        *status = SPIR_COMPUTATION_SUCCESS;
        return basis;

    } catch (...) {
        // Clean up in case of exception
        if (basis) spir_basis_release(basis);
        if (sve) spir_sve_result_release(sve);
        if (kernel) spir_kernel_release(kernel);
        *status = SPIR_INTERNAL_ERROR;
        return nullptr;
    }
}
