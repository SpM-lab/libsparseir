#pragma once

#include <memory>

#include <sparseir/sparseir.h> // C interface

// Portable helpers for C complex interoperability in C++ tests
// MSVC uses struct-based c_complex; GCC/Clang use C99 double _Complex
#ifdef _MSC_VER
#  define C_COMPLEX_MAKE(r, i) (c_complex{ (r), (i) })
#  define C_COMPLEX_REAL(z) ((z).real)
#  define C_COMPLEX_IMAG(z) ((z).imag)
#else
#  include <complex>
#  include <cstring>
   inline c_complex _c_complex_make_impl(double r, double i) {
       std::complex<double> cpp(r, i);
       c_complex c{};
       static_assert(sizeof(c_complex) == sizeof(std::complex<double>), "c_complex/std::complex size mismatch");
       std::memcpy(&c, &cpp, sizeof(c));
       return c;
   }
   inline double _c_complex_real_impl(c_complex z) {
       std::complex<double> cpp;
       std::memcpy(&cpp, &z, sizeof(z));
       return cpp.real();
   }
   inline double _c_complex_imag_impl(c_complex z) {
       std::complex<double> cpp;
       std::memcpy(&cpp, &z, sizeof(z));
       return cpp.imag();
   }
#  define C_COMPLEX_MAKE(r, i) (_c_complex_make_impl((r), (i)))
#  define C_COMPLEX_REAL(z) (_c_complex_real_impl((z)))
#  define C_COMPLEX_IMAG(z) (_c_complex_imag_impl((z)))
#endif

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
        double cutoff = -1.0;
        int lmax = -1;
        int n_gauss = -1;
        int Twork = SPIR_TWORK_FLOAT64X2;
        sve = spir_sve_result_new(kernel, epsilon, cutoff, lmax, n_gauss, Twork, &status_);
        if (status_ != SPIR_COMPUTATION_SUCCESS || sve == nullptr) {
            *status = status_;
            spir_kernel_release(kernel);
            return nullptr;
        }

        int sve_size;
        status_ = spir_sve_result_get_size(sve, &sve_size);
        if (status_ != SPIR_COMPUTATION_SUCCESS) {
            *status = status_;
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
            return nullptr;
        }
        REQUIRE(sve_size > 0);

        // Create a fermionic finite temperature basis with pre-computed SVE result
        int max_size = -1;
        basis = spir_basis_new(
            statistics, beta, omega_max, kernel, sve, max_size, &status_);
        if (status_ != SPIR_COMPUTATION_SUCCESS || basis == nullptr) {
            *status = status_;
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
            return nullptr;
        }

        int basis_size;
        status_ = spir_basis_get_size(basis, &basis_size);
        if (status_ != SPIR_COMPUTATION_SUCCESS) {
            *status = status_;
            spir_basis_release(basis);
            spir_sve_result_release(sve);
            spir_kernel_release(kernel);
        }

        std::vector<double> svals(sve_size);
        int svals_status = spir_sve_result_get_svals(sve, svals.data());
        REQUIRE(svals_status == SPIR_COMPUTATION_SUCCESS);

        REQUIRE(basis_size <= sve_size);
        if (sve_size > basis_size) {
            REQUIRE(svals[basis_size] / svals[0] <= epsilon);
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


template <typename S>
sparseir::PiecewiseLegendrePoly _singleout_poly_from_irtaufuncs(const sparseir::IRTauFuncsType<S> &funcs, std::size_t idx) {
    sparseir::IRTauFuncsType<S> funcs_slice = funcs[idx];
    sparseir::PiecewiseLegendrePolyVector u0 = funcs_slice.get_obj();
    return u0[0];
}


//template <typename S>
//sparseir::IRBasisType<S> _singleout_from_irtaufuncs(const sparseir::IRTauFuncsType<S> &funcs, std::size_t idx) {
    //sparseir::IRTauFuncsType<S> funcs_slice = funcs[idx];
//}