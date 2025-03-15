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

TEST_CASE("Kernel Accuracy Tests", "[cinterface]") {
    // Test individual kernels
    SECTION("LogisticKernel(9)") {
        auto kernel = sparseir::LogisticKernel(9);
    }

    SECTION("Kernel Domain") {
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

TEST_CASE("TauSampling", "[cinterface]") {
    SECTION("TauSampling Constructor") {
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

    SECTION("TauSampling Evaluation 1-dimensional input ROW-MAJOR") {
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

        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = dis(gen);
        }

        double *output =
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
            int status = spir_sampling_evaluate_dd(
                sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                gl_cpp_rowmajor.data(), output);

            REQUIRE(status == 0);

            // Compare results
            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<double, 4, Eigen::RowMajor> output_tensor(
                dims[0], dims[1], dims[2], dims[3]);
            for (int i = 0; i < output_tensor.size(); ++i) {
                // store output data to output_tensor
                output_tensor.data()[i] = output[i];
            }
            // Compare results
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            REQUIRE(gtau_cpp(i, j, k, l) ==
                                    output_tensor(i, j, k, l));
                        }
                    }
                }
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
    }

    SECTION("TauSampling Evaluation 4-dimensional complex input/output ROW-MAJOR")
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
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> gl_cpp_rowmajor(
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
            int status = spir_sampling_evaluate_cc(
                sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                gl_cpp_rowmajor.data(), output);

            REQUIRE(status == 0);

            // Compare results
            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> output_tensor(
                dims[0], dims[1], dims[2], dims[3]);
            for (int i = 0; i < output_tensor.size(); ++i) {
                // store output data to output_tensor
                output_tensor.data()[i] = output[i];
            }
            // Compare results
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            REQUIRE(gtau_cpp(i, j, k, l) ==
                                    output_tensor(i, j, k, l));
                        }
                    }
                }
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
    }

    SECTION("TauSampling Evaluation 4-dimensional input COLUMN-MAJOR") {
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

    SECTION("TauSampling Evaluation 4-dimensional complex input/output COLUMN-MAJOR") {
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
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp = cpp_sampling.evaluate(gl_cpp, dim);

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

TEST_CASE("MatsubaraSampling", "[cinterface]") {
    SECTION("MatsubaraSampling Constructor") {
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

    SECTION("MatsubaraSampling Evaluation 4-dimensional input COLUMN-MAJOR") {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis* basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling* sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling != nullptr);

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
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp = cpp_sampling.evaluate(gl_cpp, dim);

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
                REQUIRE(output[i].real() == Approx(gtau_cpp(i).real()));
                REQUIRE(output[i].imag() == Approx(gtau_cpp(i).imag()));
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
    }

    SECTION("MatsubaraSampling Evaluation 4-dimensional complex input COLUMN-MAJOR") {
        double beta = 1.0;
        double wmax = 10.0;

        // Create basis
        spir_fermionic_finite_temp_basis* basis = spir_fermionic_finite_temp_basis_new(beta, wmax, 1e-10);
        REQUIRE(basis != nullptr);

        // Create sampling
        spir_sampling* sampling = spir_fermionic_matsubara_sampling_new(basis);
        REQUIRE(sampling != nullptr);

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
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp = cpp_sampling.evaluate(gl_cpp, dim);

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
                REQUIRE(output[i].real() == Approx(gtau_cpp(i).real()));
                REQUIRE(output[i].imag() == Approx(gtau_cpp(i).imag()));
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
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

        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = dis(gen);
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
            Eigen::Tensor<double, 4> gl_cpp =
                sparseir::movedim(rhol_tensor, 0, dim);

            // Evaluate from real-time/tau to imaginary-time/tau
            Eigen::Tensor<std::complex<double>, 4> gtau_cpp =
                cpp_sampling.evaluate(gl_cpp, dim);
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
            int status = spir_sampling_evaluate_dc(
                sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                gl_cpp_rowmajor.data(), output);

            REQUIRE(status == 0);

            // Compare results
            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<double, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> output_tensor(
                dims[0], dims[1], dims[2], dims[3]);
            for (int i = 0; i < output_tensor.size(); ++i) {
                // store output data to output_tensor
                output_tensor.data()[i] = output[i];
            }
            // Compare results
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            REQUIRE(gtau_cpp(i, j, k, l) ==
                                    output_tensor(i, j, k, l));
                        }
                    }
                }
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
    }

    SECTION("MatsubaraSampling Evaluation 4-dimensional complex input ROW-MAJOR")
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

        // Create equivalent C++ objects for comparison
        sparseir::FiniteTempBasis<sparseir::Fermionic> cpp_basis(
            beta, wmax, 1e-10, sparseir::LogisticKernel(beta * wmax));
        sparseir::MatsubaraSampling<sparseir::Fermionic> cpp_sampling(
            cpp_basis);

        int basis_size = cpp_basis.size();

        int d1 = 2;
        int d2 = 3;
        int d3 = 4;
        Eigen::Tensor<std::complex<double>, 4> rhol_tensor(basis_size, d1, d2, d3);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < rhol_tensor.size(); ++i) {
            rhol_tensor.data()[i] = std::complex<double>(dis(gen), dis(gen));
        }

        std::complex<double> *output = (std::complex<double> *)malloc(
            basis_size * d1 * d2 * d3 * sizeof(std::complex<double>));

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
            int *dims = dims_list[dim];
            int target_dim = dim;

            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<double, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor> gl_cpp_rowmajor(
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
            int status = spir_sampling_evaluate_cc(
                sampling, SPIR_ORDER_ROW_MAJOR, ndim, dims, target_dim,
                gl_cpp_rowmajor.data(), output);

            REQUIRE(status == 0);

            // Compare results
            // Note that we need to specify Eigen::RowMajor here
            // because Eigen::Tensor<T, 4> is column-major by default
            Eigen::Tensor<std::complex<double>, 4, Eigen::RowMajor>
                output_tensor(dims[0], dims[1], dims[2], dims[3]);
            for (int i = 0; i < output_tensor.size(); ++i) {
                // store output data to output_tensor
                output_tensor.data()[i] = output[i];
            }
            // Compare results
            for (int i = 0; i < gtau_cpp.dimension(0); ++i) {
                for (int j = 0; j < gtau_cpp.dimension(1); ++j) {
                    for (int k = 0; k < gtau_cpp.dimension(2); ++k) {
                        for (int l = 0; l < gtau_cpp.dimension(3); ++l) {
                            REQUIRE(gtau_cpp(i, j, k, l) ==
                                    output_tensor(i, j, k, l));
                        }
                    }
                }
            }
        }

        // Clean up
        spir_destroy_sampling(sampling);
        spir_destroy_fermionic_finite_temp_basis(basis);
        free(output);
    }
}
