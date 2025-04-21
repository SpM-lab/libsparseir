#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <sparseir/sparseir.h>   // C interface
#include <sparseir/sparseir.hpp> // C++ interface

using Catch::Approx;
using xprec::DDouble;

TEST_CASE("Kernel Accuracy Tests", "[cinterface]")
{
    // Test individual kernels
    SECTION("LogisticKernel(9)")
    {
        auto cpp_kernel = sparseir::LogisticKernel(9);
        spir_kernel *kernel = spir_logistic_kernel_new(9);
        REQUIRE(kernel != nullptr);
    }

    SECTION("RegularizedBoseKernel(10)")
    {
        auto cpp_kernel = sparseir::RegularizedBoseKernel(10);
        spir_kernel *kernel = spir_regularized_bose_kernel_new(10);
        REQUIRE(kernel != nullptr);
    }

    SECTION("Kernel Domain")
    {
        // Create a kernel through C API
        // spir_logistic_kernel* kernel = spir_logistic_kernel_new(9);
        spir_kernel *kernel = spir_logistic_kernel_new(9);
        REQUIRE(kernel != nullptr);

        // Get domain bounds
        double xmin, xmax, ymin, ymax;
        int status = spir_kernel_domain(kernel, &xmin, &xmax, &ymin, &ymax);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);

        // Compare with C++ implementation
        auto cpp_kernel = sparseir::LogisticKernel(9);
        auto xrange = cpp_kernel.xrange();
        auto yrange = cpp_kernel.yrange();
        auto cpp_xmin = xrange.first;
        auto cpp_xmax = xrange.second;
        auto cpp_ymin = yrange.first;
        auto cpp_ymax = yrange.second;

        REQUIRE(xmin == cpp_xmin);
        REQUIRE(xmax == cpp_xmax);
        REQUIRE(ymin == cpp_ymin);
        REQUIRE(ymax == cpp_ymax);

        // Clean up
        spir_destroy_kernel(kernel);
    }
}

TEST_CASE("FiniteTempBasis", "[cinterface]")
{
    SECTION("FiniteTempBasis Constructor Fermionic")
    {
        double beta = 2.0;
        double wmax = 5.0;
        double Lambda = 10.0;
        double epsilon = 1e-6;

        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 epsilon);
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, epsilon);
        REQUIRE(basis != nullptr);
    }

    SECTION("FiniteTempBasis Constructor with SVE Fermionic/LogisticKernel")
    {
        double beta = 2.0;
        double wmax = 5.0;
        double Lambda = 10.0;
        double epsilon = 1e-6;

        spir_kernel *kernel = spir_logistic_kernel_new(Lambda);
        REQUIRE(kernel != nullptr);

        spir_sve_result *sve_result = spir_sve_result_new(kernel, epsilon);
        REQUIRE(sve_result != nullptr);

        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new_with_sve(beta, wmax, kernel,
                                                          sve_result);
        REQUIRE(basis != nullptr);

        // Clean up
        spir_destroy_kernel(kernel);
        spir_destroy_sve_result(sve_result);
        spir_destroy_fermionic_finite_temp_basis(basis);
    }

    SECTION(
        "FiniteTempBasis Constructor with SVE Bosonic/RegularizedBoseKernel")
    {
        double beta = 2.0;
        double wmax = 5.0;
        double Lambda = 10.0;
        double epsilon = 1e-6;

        spir_kernel *kernel = spir_regularized_bose_kernel_new(Lambda);
        REQUIRE(kernel != nullptr);

        spir_sve_result *sve_result = spir_sve_result_new(kernel, epsilon);
        REQUIRE(sve_result != nullptr);

        spir_bosonic_finite_temp_basis *basis =
            spir_bosonic_finite_temp_basis_new_with_sve(beta, wmax, kernel,
                                                        sve_result);
        REQUIRE(basis != nullptr);

        // Clean up
        spir_destroy_kernel(kernel);
        spir_destroy_sve_result(sve_result);
        spir_destroy_bosonic_finite_temp_basis(basis);
    }
}

TEST_CASE("DiscreteLehmannRepresentation", "[cinterface]")
{
    SECTION("DiscreteLehmannRepresentation Constructor Fermionic")
    {
        const double beta = 10000.0;
        const double wmax = 1.0;
        const double epsilon = 1e-12;

        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, epsilon);
        REQUIRE(basis != nullptr);

        spir_fermionic_dlr *dlr = spir_fermionic_dlr_new(basis);
        REQUIRE(dlr != nullptr);

        const int npoles = 10;
        Eigen::VectorXd poles(npoles);
        Eigen::VectorXd coeffs(npoles);
        std::mt19937 gen(982743);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < npoles; i++) {
            poles(i) = wmax * (2.0 * dis(gen) - 1.0);
            coeffs(i) = 2.0 * dis(gen) - 1.0;
        }
        REQUIRE(poles.array().abs().maxCoeff() <= wmax);

        spir_fermionic_dlr *dlr_with_poles =
            spir_fermionic_dlr_new_with_poles(basis, npoles, poles.data());
        REQUIRE(dlr_with_poles != nullptr);
        int fitmat_rows = spir_fermionic_dlr_fitmat_rows(dlr_with_poles);
        int fitmat_cols = spir_fermionic_dlr_fitmat_cols(dlr_with_poles);
        REQUIRE(fitmat_rows >= 0);
        REQUIRE(fitmat_cols == npoles);
        double *Gl = (double *)malloc(fitmat_rows * sizeof(double));
        int32_t to_ir_input_dims[1] = {npoles};
        int status_to_IR =
            spir_fermionic_dlr_to_IR(dlr_with_poles, SPIR_ORDER_COLUMN_MAJOR, 1,
                                     to_ir_input_dims, coeffs.data(), Gl);

        REQUIRE(status_to_IR == SPIR_COMPUTATION_SUCCESS);
        double *g_dlr = (double *)malloc(fitmat_rows * sizeof(double));
        int32_t from_ir_input_dims[1] = {static_cast<int32_t>(fitmat_rows)};
        int status_from_IR = spir_fermionic_dlr_from_IR(
            dlr, SPIR_ORDER_COLUMN_MAJOR, 1, from_ir_input_dims, Gl, g_dlr);
        REQUIRE(status_from_IR == SPIR_COMPUTATION_SUCCESS);

        // Clean up
        // free allocated memory
        free(Gl);
        free(g_dlr);

        spir_destroy_fermionic_finite_temp_basis(basis);
        spir_destroy_fermionic_dlr(dlr);
        spir_destroy_fermionic_dlr(dlr_with_poles);
    }

    SECTION("DiscreteLehmannRepresentation Constructor Bosonic")
    {
        const double beta = 10000.0;
        const double wmax = 1.0;
        const double epsilon = 1e-12;

        spir_bosonic_finite_temp_basis *basis =
            spir_bosonic_finite_temp_basis_new(beta, wmax, epsilon);
        REQUIRE(basis != nullptr);

        spir_bosonic_dlr *dlr = spir_bosonic_dlr_new(basis);
        REQUIRE(dlr != nullptr);

        const int npoles = 10;
        Eigen::VectorXd poles(npoles);
        Eigen::VectorXd coeffs(npoles);
        std::mt19937 gen(982743);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < npoles; i++) {
            poles(i) = wmax * (2.0 * dis(gen) - 1.0);
            coeffs(i) = 2.0 * dis(gen) - 1.0;
        }
        REQUIRE(poles.array().abs().maxCoeff() <= wmax);

        spir_bosonic_dlr *dlr_with_poles =
            spir_bosonic_dlr_new_with_poles(basis, npoles, poles.data());
        REQUIRE(dlr_with_poles != nullptr);
        int fitmat_rows = spir_bosonic_dlr_fitmat_rows(dlr_with_poles);
        int fitmat_cols = spir_bosonic_dlr_fitmat_cols(dlr_with_poles);
        REQUIRE(fitmat_rows >= 0);
        REQUIRE(fitmat_cols == npoles);
        double *Gl = (double *)malloc(fitmat_rows * sizeof(double));
        int32_t to_ir_input_dims[1] = {npoles};
        int status_to_IR =
            spir_bosonic_dlr_to_IR(dlr_with_poles, SPIR_ORDER_COLUMN_MAJOR, 1,
                                   to_ir_input_dims, coeffs.data(), Gl);

        REQUIRE(status_to_IR == SPIR_COMPUTATION_SUCCESS);
        double *g_dlr = (double *)malloc(fitmat_rows * sizeof(double));
        int32_t from_ir_input_dims[1] = {static_cast<int32_t>(fitmat_rows)};
        int status_from_IR = spir_bosonic_dlr_from_IR(
            dlr, SPIR_ORDER_COLUMN_MAJOR, 1, from_ir_input_dims, Gl, g_dlr);
        REQUIRE(status_from_IR == SPIR_COMPUTATION_SUCCESS);

        // Clean up
        // free allocated memory
        free(Gl);
        free(g_dlr);

        spir_destroy_bosonic_finite_temp_basis(basis);
        spir_destroy_bosonic_dlr(dlr);
        spir_destroy_bosonic_dlr(dlr_with_poles);
    }
}

TEST_CASE("TauSampling", "[cinterface]")
{
    SECTION("TauSampling Constructor (fermionic)")
    {
        double beta = 1.0;
        double wmax = 10.0;

        auto basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-15);
        REQUIRE(basis != nullptr);

        auto sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Test getting sampling points
        double *tau_points = (double *)malloc(n_points * sizeof(double));
        int status_get_tau_points = spir_sampling_get_tau_points(sampling, tau_points);
        REQUIRE(status_get_tau_points == SPIR_COMPUTATION_SUCCESS);
        free(tau_points);

        int *matsubara_points = (int *)malloc(n_points * sizeof(int));
        status = spir_sampling_get_matsubara_points(sampling, matsubara_points);
        REQUIRE(status == SPIR_NOT_SUPPORTED);
        free(matsubara_points);

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
    }

    SECTION("TauSampling Constructor (bosonic)")
    {
        double beta = 1.0;
        double wmax = 10.0;

        auto basis = spir_bosonic_finite_temp_basis_new(beta, wmax, 1e-15);
        REQUIRE(basis != nullptr);

        auto sampling = spir_bosonic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Test getting sampling points
        double *tau_points = (double *)malloc(n_points * sizeof(double));
        int status_get_tau_points = spir_sampling_get_tau_points(sampling, tau_points);
        REQUIRE(status_get_tau_points == SPIR_COMPUTATION_SUCCESS);
        free(tau_points);

        int *matsubara_points = (int *)malloc(n_points * sizeof(int));
        status = spir_sampling_get_matsubara_points(sampling, matsubara_points);
        REQUIRE(status == SPIR_NOT_SUPPORTED);
        free(matsubara_points);

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_bosonic_finite_temp_basis(basis);
    }

    SECTION("TauSampling Evaluation 1-dimensional input ROW-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();
        Eigen::VectorXd cpp_Gl_vec = Eigen::VectorXd::Random(basis_size);
        Eigen::Tensor<double, 1> cpp_Gl(basis_size);
        for (size_t i = 0; i < basis_size; ++i) {
            cpp_Gl(i) = cpp_Gl_vec(i);
        }
        Eigen::Tensor<double, 1> Gtau_cpp = cpp_sampling.evaluate(cpp_Gl);
        Eigen::Tensor<double, 1> gl_from_tau = cpp_sampling.fit(Gtau_cpp);

        // Set up parameters for evaluation
        int ndim = 1;
        int dims[1] = {basis_size};
        int target_dim = 0;

        // Allocate memory for coefficients
        double *coeffs = (double *)malloc(basis_size * sizeof(double));
        // Create coefficients (simple test values)
        for (int i = 0; i < basis_size; i++) {
            coeffs[i] = cpp_Gl_vec(i);
        }

        // Create output buffer
        double *evaluate_output = (double *)malloc(basis_size * sizeof(double));
        double *fit_output = (double *)malloc(basis_size * sizeof(double));

        // Evaluate using C API
        int evaluate_status = spir_sampling_evaluate_dd(
            sampling,
            SPIR_ORDER_ROW_MAJOR, // Assuming this enum is defined in the header
            ndim, dims, target_dim, coeffs, evaluate_output);

        REQUIRE(evaluate_status == 0);

        for (int i = 0; i < basis_size; i++) {
            REQUIRE(evaluate_output[i] == Approx(Gtau_cpp(i)));
        }

        int fit_status = spir_sampling_fit_dd(
            sampling,
            SPIR_ORDER_COLUMN_MAJOR, // Assuming this enum is defined in the
                                     // header
            ndim, dims, target_dim, evaluate_output, fit_output);

        REQUIRE(fit_status == 0);

        for (int i = 0; i < basis_size; i++) {
            REQUIRE(fit_output[i] == Approx(gl_from_tau(i)));
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        // Free allocated memory
        free(coeffs);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION("TauSampling Evaluation 1-dimensional input COLUMN-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();
        Eigen::VectorXd cpp_Gl_vec = Eigen::VectorXd::Random(basis_size);
        Eigen::Tensor<double, 1> cpp_Gl(basis_size);
        for (size_t i = 0; i < basis_size; ++i) {
            cpp_Gl(i) = cpp_Gl_vec(i);
        }
        Eigen::Tensor<double, 1> Gtau_cpp = cpp_sampling.evaluate(cpp_Gl);
        Eigen::Tensor<double, 1> gl_from_tau = cpp_sampling.fit(Gtau_cpp);

        // Set up parameters for evaluation
        int ndim = 1;
        int dims[1] = {basis_size};
        int target_dim = 0;

        // Allocate memory for coefficients
        double *coeffs = (double *)malloc(basis_size * sizeof(double));
        // Create coefficients (simple test values)
        for (int i = 0; i < basis_size; i++) {
            coeffs[i] = cpp_Gl_vec(i);
        }

        // Create output buffer
        double *evaluate_output = (double *)malloc(basis_size * sizeof(double));
        double *fit_output = (double *)malloc(basis_size * sizeof(double));

        // Evaluate using C API
        int evaluate_status = spir_sampling_evaluate_dd(
            sampling,
            SPIR_ORDER_COLUMN_MAJOR, // Assuming this enum is defined in the
                                     // header
            ndim, dims, target_dim, coeffs, evaluate_output);

        REQUIRE(evaluate_status == 0);

        for (int i = 0; i < basis_size; i++) {
            REQUIRE(evaluate_output[i] == Approx(Gtau_cpp(i)));
        }

        int fit_status = spir_sampling_fit_dd(
            sampling,
            SPIR_ORDER_COLUMN_MAJOR, // Assuming this enum is defined in the
                                     // header
            ndim, dims, target_dim, evaluate_output, fit_output);

        REQUIRE(fit_status == 0);

        for (int i = 0; i < basis_size; i++) {
            REQUIRE(fit_output[i] == Approx(gl_from_tau(i)));
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        // Free allocated memory
        free(coeffs);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION("TauSampling Evaluation 4-dimensional input ROW-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = dis(gen);
        }

        double *evaluate_output =
            (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));
        double *fit_output =
            (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<double, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<double, 4> gtau_cpp =
                cpp_sampling.evaluate(gl_cpp, dim);

            Eigen::Tensor<double, 4> gl_cpp_fit =
                cpp_sampling.fit(gtau_cpp, dim);

            int *dims = dims_list[dim];
            int target_dim = dim;

            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<double, 4, Eigen::RowMajor> gl_cpp_rowmajor(
                dims[0], dims[1], dims[2], dims[3]);

            // Fill row-major tensor in the correct order
            for (int i = 0; i < gl_cpp.dimension(0); ++i) {
                for (int j = 0; j < gl_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gl_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gl_cpp.dimension(3); ++l) {
                            gl_cpp_rowmajor(i, j, k, l) = gl_cpp(i, j, k, l);
                        }
                    }
                }
            }
            // Evaluate using C API
            int evaluate_status = spir_sampling_evaluate_dd(
                sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                gl_cpp_rowmajor.data(), evaluate_output);

            REQUIRE(evaluate_status == 0);

            int fit_status =
                spir_sampling_fit_dd(sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims,
                                     target_dim, evaluate_output, fit_output);

            REQUIRE(fit_status == 0);

            // Compare results
            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<double, 4, Eigen::RowMajor> output_tensor(
                dims[0], dims[1], dims[2], dims[3]);
            Eigen::Tensor<double, 4, Eigen::RowMajor> fit_tensor(
                dims[0], dims[1], dims[2], dims[3]);
            for (int i = 0; i < output_tensor.size(); ++i) {
                // store output data to output_tensor
                output_tensor.data()[i] = evaluate_output[i];
                fit_tensor.data()[i] = fit_output[i];
            }
            // Compare results
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            REQUIRE(gtau_cpp(i, j, k, l) ==
                                    output_tensor(i, j, k, l));
                            REQUIRE(gl_cpp_fit(i, j, k, l) ==
                                    fit_tensor(i, j, k, l));
                        }
                    }
                }
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION(
        "TauSampling Evaluation 4-dimensional complex input/output ROW-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2,
                                                           d3);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
        }

        c_complex *evaluate_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        c_complex *fit_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<std::complex<double>, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
                cpp_sampling.evaluate(gl_cpp, dim);
            Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
                cpp_sampling.fit(gtau_cpp, dim);
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor>
                gl_cpp_rowmajor(dims[0], dims[1], dims[2], dims[3]);

            // Fill row-major tensor in the correct order
            for (int i = 0; i < gl_cpp.dimension(0); ++i) {
                for (int j = 0; j < gl_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gl_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gl_cpp.dimension(3); ++l) {
                            gl_cpp_rowmajor(i, j, k, l) = gl_cpp(i, j, k, l);
                        }
                    }
                }
            }

            c_complex *evaluate_input = (c_complex *)malloc(
                basis_size * d1 * d2 * d3 * sizeof(c_complex));
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                evaluate_input[i] =
                    (c_complex){gl_cpp_rowmajor.data()[i].real(),
                                gl_cpp_rowmajor.data()[i].imag()};
            }
            // Evaluate using C API
            int evaluate_status = spir_sampling_evaluate_zz(
                sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                evaluate_input, evaluate_output);

            REQUIRE(evaluate_status == 0);

            int fit_status =
                spir_sampling_fit_zz(sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims,
                                     target_dim, evaluate_output, fit_output);

            REQUIRE(fit_status == 0);

            // Compare results
            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor>
                output_tensor(dims[0], dims[1], dims[2], dims[3]);
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> fit_tensor(
                dims[0], dims[1], dims[2], dims[3]);
            for (int i = 0; i < output_tensor.size(); ++i) {
                // store output data to output_tensor
                // to std::complex<double>
                output_tensor.data()[i] = (std::complex<double>){
                    __real__ evaluate_output[i], __imag__ evaluate_output[i]};
                fit_tensor.data()[i] = (std::complex<double>){
                    __real__ fit_output[i], __imag__ fit_output[i]};
            }
            // Compare results
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            REQUIRE(gtau_cpp(i, j, k, l) ==
                                    output_tensor(i, j, k, l));
                            REQUIRE(gl_cpp_fit(i, j, k, l) ==
                                    fit_tensor(i, j, k, l));
                        }
                    }
                }
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION("TauSampling Evaluation 4-dimensional input COLUMN-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        // Fill rhol_tensor with random complex values
        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = dis(gen);
        }

        double *evaluate_output =
            (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));
        double *fit_output =
            (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<double, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<double, 4> gtau_cpp =
                cpp_sampling.evaluate(gl_cpp, dim);
            Eigen::Tensor<double, 4> gl_cpp_fit =
                cpp_sampling.fit(gtau_cpp, dim);

            // Set up parameters for evaluation
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Evaluate using C API
            int evaluate_status = spir_sampling_evaluate_dd(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                gl_cpp.data(), evaluate_output);

            REQUIRE(evaluate_status == 0);

            int fit_status = spir_sampling_fit_dd(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                evaluate_output, fit_output);

            REQUIRE(fit_status == 0);
            // Compare with C++ implementation
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                REQUIRE(evaluate_output[i] == gtau_cpp(i));
                // TODO: fix this
                REQUIRE(gl_cpp_fit(i) == Approx(gl_cpp(i)));
                REQUIRE(fit_output[i] == gl_cpp_fit(i));
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION("TauSampling Evaluation 4-dimensional complex input/output "
            "COLUMN-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2,
                                                           d3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        // Fill rhol_tensor with random complex values
        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
        }

        c_complex *evaluate_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        c_complex *fit_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<std::complex<double>, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);
            c_complex *evaluate_input = (c_complex *)malloc(
                basis_size * d1 * d2 * d3 * sizeof(c_complex));
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                evaluate_input[i] = (c_complex){gl_cpp.data()[i].real(),
                                                gl_cpp.data()[i].imag()};
            }
            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
                cpp_sampling.evaluate(gl_cpp, dim);
            Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
                cpp_sampling.fit(gtau_cpp, dim);
            // Set up parameters for evaluation
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Evaluate using C API
            int evaluate_status = spir_sampling_evaluate_zz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                evaluate_input, evaluate_output);
            REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

            int fit_status = spir_sampling_fit_zz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                evaluate_output, fit_output);
            REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

            // Compare with C++ implementation
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                REQUIRE(__real__ evaluate_output[i] ==
                        Approx(gtau_cpp(i).real()));
                REQUIRE(__imag__ evaluate_output[i] ==
                        Approx(gtau_cpp(i).imag()));
                REQUIRE(__real__ fit_output[i] == Approx(gl_cpp_fit(i).real()));
                REQUIRE(__imag__ fit_output[i] == Approx(gl_cpp_fit(i).imag()));
            }
            free(evaluate_input);
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION("TauSampling Error Status")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        // Fill rhol_tensor with random complex values
        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = dis(gen);
        }

        c_complex *output_complex =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        double *output_double =
            (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<double, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Set up parameters for evaluation
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Evaluate using C API that is not supported
            int status_not_supported = spir_sampling_evaluate_dz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                gl_cpp.data(), output_complex);
            REQUIRE(status_not_supported == SPIR_NOT_SUPPORTED);

            if (dim == 0) {
                continue;
            }

            // Evaluate using C API that has dimension mismatch
            int evaluate_status_dimension_mismatch = spir_sampling_evaluate_dd(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims1, target_dim,
                gl_cpp.data(), output_double);
            REQUIRE(evaluate_status_dimension_mismatch ==
                    SPIR_INPUT_DIMENSION_MISMATCH);

            int fit_status_dimension_mismatch = spir_sampling_fit_dd(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims1, target_dim,
                output_double, output_double);
            REQUIRE(fit_status_dimension_mismatch ==
                    SPIR_INPUT_DIMENSION_MISMATCH);
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output_complex);
        free(output_double);
    }
}

TEST_CASE("MatsubaraSampling", "[cinterface]")
{
    SECTION("MatsubaraSampling Constructor")
    {
        double beta = 1.0;
        double wmax = 10.0;

        auto basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-15);
        REQUIRE(basis != nullptr);

        auto sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling != nullptr);
        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
    }

    SECTION("MatsubaraSampling Evaluation 4-dimensional input COLUMN-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::MatsubaraSampling<sparseir::Fermionic> cpp_sampling(
            cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        // Fill rhol_tensor with random complex values
        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = dis(gen);
        }

        c_complex *evaluate_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        c_complex *fit_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<double, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
                cpp_sampling.evaluate(gl_cpp, dim);
            Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
                cpp_sampling.fit(gtau_cpp, dim);
            // Set up parameters for evaluation
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Evaluate using C API
            int evaluate_status = spir_sampling_evaluate_dz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                gl_cpp.data(), evaluate_output);
            REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

            int fit_status = spir_sampling_fit_zz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                evaluate_output, fit_output);
            REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

            // Compare with C++ implementation
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                REQUIRE(__real__ evaluate_output[i] ==
                        Approx(gtau_cpp(i).real()));
                REQUIRE(__imag__ evaluate_output[i] ==
                        Approx(gtau_cpp(i).imag()));
                REQUIRE(__real__ fit_output[i] == Approx(gl_cpp_fit(i).real()));
                REQUIRE(__imag__ fit_output[i] == Approx(gl_cpp_fit(i).imag()));
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION(
        "MatsubaraSampling Evaluation 4-dimensional complex input COLUMN-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::MatsubaraSampling<sparseir::Fermionic> cpp_sampling(
            cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2,
                                                           d3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        // Fill rhol_tensor with random complex values
        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
        }

        c_complex *evaluate_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        c_complex *fit_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<std::complex<double>, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
                cpp_sampling.evaluate(gl_cpp, dim);
            Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
                cpp_sampling.fit(gtau_cpp, dim);
            // Set up parameters for evaluation
            int *dims = dims_list[dim];
            int target_dim = dim;

            c_complex *evaluate_input = (c_complex *)malloc(
                basis_size * d1 * d2 * d3 * sizeof(c_complex));
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                evaluate_input[i] = (c_complex){gl_cpp.data()[i].real(),
                                                gl_cpp.data()[i].imag()};
            }
            // Evaluate using C API
            int evaluate_status = spir_sampling_evaluate_zz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                evaluate_input, evaluate_output);
            REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

            int fit_status = spir_sampling_fit_zz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                evaluate_output, fit_output);
            REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

            // Compare with C++ implementation
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                REQUIRE(__real__ evaluate_output[i] ==
                        Approx(gtau_cpp(i).real()));
                REQUIRE(__imag__ evaluate_output[i] ==
                        Approx(gtau_cpp(i).imag()));
                REQUIRE(__real__ fit_output[i] == Approx(gl_cpp_fit(i).real()));
                REQUIRE(__imag__ fit_output[i] == Approx(gl_cpp_fit(i).imag()));
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION("MatsubaraSampling Evaluation 4-dimensional input ROW-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::MatsubaraSampling<sparseir::Fermionic> cpp_sampling(
            cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = dis(gen);
        }

        c_complex *evaluate_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        c_complex *fit_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<double, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
                cpp_sampling.evaluate(gl_cpp, dim);
            Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
                cpp_sampling.fit(gtau_cpp, dim);
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<double, 4, Eigen::RowMajor> gl_cpp_rowmajor(
                dims[0], dims[1], dims[2], dims[3]);

            // Fill row-major tensor in the correct order
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            gl_cpp_rowmajor(i, j, k, l) = gl_cpp(i, j, k, l);
                        }
                    }
                }
            }
            // Evaluate using C API
            int evaluate_status = spir_sampling_evaluate_dz(
                sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                gl_cpp_rowmajor.data(), evaluate_output);
            REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

            int fit_status =
                spir_sampling_fit_zz(sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims,
                                     target_dim, evaluate_output, fit_output);
            REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

            // Compare results
            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<double, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor>
                output_tensor(dims[0], dims[1], dims[2], dims[3]);
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> fit_tensor(
                dims[0], dims[1], dims[2], dims[3]);
            for (int i = 0; i < output_tensor.size(); ++i) {
                // store output data to output_tensor
                output_tensor.data()[i] = std::complex<double>(
                    __real__ evaluate_output[i], __imag__ evaluate_output[i]);
                fit_tensor.data()[i] = std::complex<double>(
                    __real__ fit_output[i], __imag__ fit_output[i]);
            }
            // Compare results
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            REQUIRE(gtau_cpp(i, j, k, l) ==
                                    output_tensor(i, j, k, l));
                            REQUIRE(gl_cpp_fit(i, j, k, l) ==
                                    fit_tensor(i, j, k, l));
                        }
                    }
                }
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION(
        "MatsubaraSampling Evaluation 4-dimensional complex input ROW-MAJOR")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::MatsubaraSampling<sparseir::Fermionic> cpp_sampling(
            cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2,
                                                           d3);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
        }

        c_complex *evaluate_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));
        c_complex *fit_output =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<std::complex<double>, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);
            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
                cpp_sampling.evaluate(gl_cpp, dim);
            Eigen::Tensor<std::complex<double>, 4> gl_cpp_fit =
                cpp_sampling.fit(gtau_cpp, dim);
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<double, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor>
                gl_cpp_rowmajor(dims[0], dims[1], dims[2], dims[3]);

            // Fill row-major tensor in the correct order
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            gl_cpp_rowmajor(i, j, k, l) = gl_cpp(i, j, k, l);
                        }
                    }
                }
            }
            c_complex *evaluate_input = (c_complex *)malloc(
                basis_size * d1 * d2 * d3 * sizeof(c_complex));
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                evaluate_input[i] =
                    (c_complex){gl_cpp_rowmajor.data()[i].real(),
                                gl_cpp_rowmajor.data()[i].imag()};
            }
            // Evaluate using C API
            int evaluate_status = spir_sampling_evaluate_zz(
                sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                evaluate_input, evaluate_output);
            REQUIRE(evaluate_status == SPIR_COMPUTATION_SUCCESS);

            int fit_status =
                spir_sampling_fit_zz(sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims,
                                     target_dim, evaluate_output, fit_output);
            REQUIRE(fit_status == SPIR_COMPUTATION_SUCCESS);

            // Compare results
            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor>
                output_tensor(dims[0], dims[1], dims[2], dims[3]);
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> fit_tensor(
                dims[0], dims[1], dims[2], dims[3]);
            for (int i = 0; i < output_tensor.size(); ++i) {
                // store output data to output_tensor
                output_tensor.data()[i] = std::complex<double>(
                    __real__ evaluate_output[i], __imag__ evaluate_output[i]);
                fit_tensor.data()[i] = std::complex<double>(
                    __real__ fit_output[i], __imag__ fit_output[i]);
            }
            // Compare results
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            REQUIRE(gtau_cpp(i, j, k, l) ==
                                    output_tensor(i, j, k, l));
                            REQUIRE(gl_cpp_fit(i, j, k, l) ==
                                    fit_tensor(i, j, k, l));
                        }
                    }
                }
            }
            free(evaluate_input);
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(evaluate_output);
        free(fit_output);
    }

    SECTION("MatsubaraSampling Error Status")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis *basis =
            spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling *sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Test getting number of sampling points
        int n_points;
        int status = spir_sampling_get_num_points(sampling, &n_points);
        REQUIRE(status == SPIR_COMPUTATION_SUCCESS);
        REQUIRE(n_points > 0);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(beta, wmax,
                                                                 1e-10);
        sparseir::MatsubaraSampling<sparseir::Fermionic> cpp_sampling(
            cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<double, 4> rhol_tensor(basis_size, d1, d2, d3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        // Fill rhol_tensor with random complex values
        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = dis(gen);
        }

        // std::complex<double>* output_complex =
        // (std::complex<double>*)malloc(basis_size * d1 * d2 * d3 *
        // sizeof(std::complex<double>));
        c_complex *output_complex =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        double *output_double =
            (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));
        double *fit_output_double =
            (double *)malloc(basis_size * d1 * d2 * d3 * sizeof(double));
        c_complex *fit_output_complex =
            (c_complex *)malloc(basis_size * d1 * d2 * d3 * sizeof(c_complex));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int *> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<double, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Set up parameters for evaluation
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Evaluate using C API that is not supported
            int status_not_supported = spir_sampling_evaluate_dd(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                gl_cpp.data(), output_double);
            REQUIRE(status_not_supported == SPIR_NOT_SUPPORTED);

            int fit_status_not_supported = spir_sampling_fit_dd(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim,
                output_double, fit_output_double);
            REQUIRE(fit_status_not_supported == SPIR_NOT_SUPPORTED);

            if (dim == 0) {
                continue;
            }

            // Evaluate using C API that has dimension mismatch
            int status_dimension_mismatch = spir_sampling_evaluate_dz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims1, target_dim,
                gl_cpp.data(), output_complex);
            REQUIRE(status_dimension_mismatch == SPIR_INPUT_DIMENSION_MISMATCH);

            int fit_status_dimension_mismatch = spir_sampling_fit_zz(
                sampling, SPIR_ORDER_COLUMN_MAJOR, ndim, dims1, target_dim,
                output_complex, fit_output_complex);
            REQUIRE(fit_status_dimension_mismatch ==
                    SPIR_INPUT_DIMENSION_MISMATCH);
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output_complex);
        free(output_double);
        free(fit_output_double);
        free(fit_output_complex);
    }
}
