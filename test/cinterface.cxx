#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>
#include <catch2/catch_approx.hpp>

#include <sparseir/sparseir.h> // C interface
#include <sparseir/sparseir.hpp> // C++ interface

using Catch::Approx;
using xprec::DDouble;

TEST_CASE("Kernel Accuracy Tests", "[cinterface]")
{
    // Test individual kernels
    SECTION("LogisticKernel(9)")
    {
        auto kernel = sparseir::LogisticKernel(9);
    }

    SECTION("Kernel Domain")
    {
        // Create a kernel through C API
        //spir_logistic_kernel* kernel = spir_logistic_kernel_new(9);
        spir_kernel* kernel = spir_logistic_kernel_new(9);
        REQUIRE(kernel != nullptr);

        // Get domain bounds
        double xmin, xmax, ymin, ymax;
        int status = spir_kernel_domain(kernel, &xmin, &xmax, &ymin, &ymax);
        REQUIRE(status == 0);

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

TEST_CASE("TauSampling", "[cinterface]")
{
    SECTION("TauSampling Constructor")
    {
        double beta = 1.0;
        double wmax = 10.0;

        auto basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-15);
        REQUIRE(basis != nullptr);

        auto sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);
        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
    }

    SECTION("TauSampling Evaluation 1-dimensional input")
    {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis* basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling* sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling != nullptr);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(
            beta, wmax, 1e-10, sparseir::LogisticKernel(beta * wmax));
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();
        Eigen::VectorXd cpp_Gl_vec = Eigen::VectorXd::Random(basis_size);
        Eigen::Tensor<double, 1> cpp_Gl(basis_size);
        for (size_t i = 0; i < basis_size; ++i) {
            cpp_Gl(i) = cpp_Gl_vec(i);
        }
        Eigen::Tensor<double, 1> Gtau_cpp = cpp_sampling.evaluate(cpp_Gl);

        // Set up parameters for evaluation
        int ndim = 1;
        int dims[1] = {basis_size};
        int target_dim = 0;

        // Allocate memory for coefficients
        double* coeffs = (double*)malloc(basis_size * sizeof(double));
        // Create coefficients (simple test values)
        for (int i = 0; i < basis_size; i++) {
            coeffs[i] = cpp_Gl_vec(i);
        }

        // Create output buffer
        double* output = (double*)malloc(basis_size * sizeof(double));

        // Evaluate using C API
        int status = spir_sampling_evaluate_dd(
            sampling,
            SPIR_ORDER_ROW_MAJOR,  // Assuming this enum is defined in the header
            ndim,
            dims,
            target_dim,
            coeffs,
            output
        );

        REQUIRE(status == 0);

        for (int i = 0; i < basis_size; i++) {
            REQUIRE(output[i] == Approx(Gtau_cpp(i)));
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        // Free allocated memory
        free(coeffs);
        free(output);
    }

    TEST_CASE("TauSampling_debug", "[cinterface]")
    {
        SECTION("TauSampling Evaluation 2-dimensional input ROW-MAJOR",
                "[cinterface]")
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

            // Create equivalent C++ objects for comparison
            sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(
                beta, wmax, 1e-10, sparseir::LogisticKernel(beta * wmax));
            sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

            int basis_size = cpp_basis.size();

            int d1 = 2;
            int ndim = 2;

            Eigen::Tensor<double, 2> gl_cpp(basis_size, d1);
            REQUIRE(basis_size == 14);
            // 14 x d1
            double input[28] = {0.2738, 0.5183, 0.9021, 0.1847, 0.6451, 0.3379,
                                0.1156, 0.7972, 0.4850, 0.2301, 0.6123, 0.9630,
                                0.0417, 0.8888, 0.5023, 0.3190, 0.1746, 0.7325,
                                0.6574, 0.0192, 0.4512, 0.2843, 0.9056, 0.1324,
                                0.7765, 0.8437, 0.0689, 0.5555};

            for (int i = 0; i < 28; ++i) {
                gl_cpp.data()[i] = input[i];
            }

            double *output = (double *)malloc(basis_size * d1 * sizeof(double));

            // Evaluate from real-time/tau to imaginary-time/tau
            {
                int dims[2] = {basis_size, d1};
                int target_dim = 0;
                Eigen::Tensor<double, 2> gtau_cpp =
                    cpp_sampling.evaluate(gl_cpp, target_dim);

                // C-API with row-major input
                Eigen::Tensor<double, 2, Eigen::RowMajor> gl_cpp_rowmajor(
                    basis_size, d1);

                // Fill row-major tensor in the correct order
                for (int i = 0; i < basis_size; ++i) {
                    for (int j = 0; j < d1; ++j) {
                        gl_cpp_rowmajor(i, j) = gl_cpp(i, j);
                    }
                }

                // Evaluate using C API
                int status = spir_sampling_evaluate_dd(
                    sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                    gl_cpp_rowmajor.data(), output);

                REQUIRE(status == 0);

                for (int i = 0; i < basis_size; ++i) {
                    for (int j = 0; j < d1; ++j) {
                        REQUIRE(gtau_cpp(i, j) == output[i * d1 + j]);
                    }
                }
            }

            // Evaluate from real-time/tau to imaginary-time/tau
            {
                int dims[2] = {d1, basis_size};
                int target_dim = 1;
                gl_cpp = sparseir::movedim(gl_cpp, 0, target_dim);
                Eigen::Tensor<double, 2> gtau_cpp =
                    cpp_sampling.evaluate(gl_cpp, target_dim);

                // C-API with row-major input
                Eigen::Tensor<double, 2, Eigen::RowMajor> gl_cpp_rowmajor(
                    d1, basis_size);

                // Fill row-major tensor in the correct order
                for (int i = 0; i < d1; ++i) {
                    for (int j = 0; j < basis_size; ++j) {
                        gl_cpp_rowmajor(i, j) = gl_cpp(i, j);
                    }
                }

                // Evaluate using C API
                int status = spir_sampling_evaluate_dd(
                    sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                    gl_cpp_rowmajor.data(), output);

                REQUIRE(status == 0);

                for (int i = 0; i < d1; ++i) {
                    for (int j = 0; j < basis_size; ++j) {
                        REQUIRE(gtau_cpp(i, j) == output[i * basis_size + j]);
                    }
                }
            }

            // Clean up
            spir_destroy_sampling(sampling);
            spir_destroy_fermionic_finite_temp_basis(basis);
            free(output);
        }
    }

    SECTION("TauSampling Evaluation 4-dimensional input COLUMN-MAJOR", "[cinterface]"){
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis* basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis!= nullptr);

        // Create sampling
        spir_sampling* sampling = spir_fermionic_tau_sampling_new(basis);
        REQUIRE(sampling!= nullptr);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(
            beta, wmax, 1e-10, sparseir::LogisticKernel(beta * wmax));
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

        double* output = (double*)malloc(basis_size * d1 * d2 * d3 * sizeof(double));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int*> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<double, 4> gl_cpp = sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<double, 4> gtau_cpp = cpp_sampling.evaluate(gl_cpp, dim);

            // Set up parameters for evaluation
            int* dims = dims_list[dim];
            int target_dim = dim;

            // Evaluate using C API
            int status = spir_sampling_evaluate_dd(
                sampling,
                SPIR_ORDER_COLUMN_MAJOR,
                ndim,
                dims,
                target_dim,
                gl_cpp.data(),
                output
            );
            REQUIRE(status == 0);

            // Compare with C++ implementation
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                REQUIRE(output[i] == Approx(gtau_cpp(i)));
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
    }

    SECTION("TauSampling Evaluation 4-dimensional complex input COLUMN-MAJOR",
            "[cinterface]")
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

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(
            beta, wmax, 1e-10, sparseir::LogisticKernel(beta * wmax));
        sparseir::TauSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2, d3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        // Fill rhol_tensor with random complex values
        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
        }

        std::complex<double> *output =
            (std::complex<double> *)malloc(basis_size * d1 * d2 * d3 * sizeof(std::complex<double>));

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

            // Set up parameters for evaluation
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Evaluate using C API
            int status = spir_sampling_evaluate_cc(
                sampling,
                SPIR_ORDER_COLUMN_MAJOR,
                ndim, dims, target_dim, gl_cpp.data(), output);
            REQUIRE(status == 0);

            // Compare with C++ implementation
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                REQUIRE(output[i].real() == Approx(gtau_cpp(i).real()));
                REQUIRE(output[i].imag() == Approx(gtau_cpp(i).imag()));
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
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

    SECTION("MatsubaraSampling Evaluation 4-dimensional input COLUMN-MAJOR", "[cinterface]"){
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis* basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis!= nullptr);

        // Create sampling
        spir_sampling* sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling!= nullptr);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(
            beta, wmax, 1e-10, sparseir::LogisticKernel(beta * wmax));
        sparseir::MatsubaraSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

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

        std::complex<double>* output = (std::complex<double>*)malloc(basis_size * d1 * d2 * d3 * sizeof(std::complex<double>));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int*> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<double, 4> gl_cpp = sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<std::complex<double>, 4> gmats_cpp = cpp_sampling.evaluate(gl_cpp, dim);

            // Set up parameters for evaluation
            int* dims = dims_list[dim];
            int target_dim = dim;

            // Evaluate using C API
            int status = spir_sampling_evaluate_dc(
                sampling,
                SPIR_ORDER_COLUMN_MAJOR,
                ndim,
                dims,
                target_dim,
                gl_cpp.data(),
                output
            );
            REQUIRE(status == 0);

            // Compare with C++ implementation
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                REQUIRE(output[i].real() == Approx(gmats_cpp(i).real()));
                REQUIRE(output[i].imag() == Approx(gmats_cpp(i).imag()));
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
    }

    SECTION("MatsubaraSampling Evaluation 4-dimensional complex input COLUMN-MAJOR", "[cinterface]"){
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis* basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis!= nullptr);

        // Create sampling
        spir_sampling* sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling!= nullptr);

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(
            beta, wmax, 1e-10, sparseir::LogisticKernel(beta * wmax));
        sparseir::MatsubaraSampling<sparseir::Fermionic> cpp_sampling(cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2, d3);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        // Fill rhol_tensor with random complex values
        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
        }

        std::complex<double>* output = (std::complex<double>*)malloc(basis_size * d1 * d2 * d3 * sizeof(std::complex<double>));

        int ndim = 4;
        int dims1[4] = {basis_size, d1, d2, d3};
        int dims2[4] = {d1, basis_size, d2, d3};
        int dims3[4] = {d1, d2, basis_size, d3};
        int dims4[4] = {d1, d2, d3, basis_size};

        std::vector<int*> dims_list = {dims1, dims2, dims3, dims4};

        // Test evaluate() and fit() along each dimension
        for (int dim = 0; dim < 4; ++dim) {
            // Move the "frequency" dimension around
            // julia> gl = SparseIR.movedim(originalgl, 1 => dim)
            Eigen::Tensor<std::complex<double>, 4> gl_cpp = sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<std::complex<double>, 4> gmats_cpp = cpp_sampling.evaluate(gl_cpp, dim);

            // Set up parameters for evaluation
            int* dims = dims_list[dim];
            int target_dim = dim;

            // Evaluate using C API
            int status = spir_sampling_evaluate_cc(
                sampling,
                SPIR_ORDER_COLUMN_MAJOR,
                ndim,
                dims,
                target_dim,
                gl_cpp.data(),
                output
            );
            REQUIRE(status == 0);

            // Compare with C++ implementation
            for (int i = 0; i < basis_size * d1 * d2 * d3; ++i) {
                REQUIRE(output[i].real() == Approx(gmats_cpp(i).real()));
                REQUIRE(output[i].imag() == Approx(gmats_cpp(i).imag()));
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
    }
}
