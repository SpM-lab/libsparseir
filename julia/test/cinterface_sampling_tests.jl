# This file corresponds to test/cpp/cinterface_integration.cxx
# Multi-dimensional tensor operations test covering different dimensions,
# target dimensions, and storage orders.

@testitem "tensor" begin
    using LibSparseIR

    β = 10.0
    ωmax = 2.0
    ε = 1e-10
    tol = 10 * ε

    # Helper function to generate multi-dimensional tensor dimensions
    function get_dims(target_dim_size, extra_dims, target_dim, ndim)
        dims = zeros(Int32, ndim)
        dims[target_dim + 1] = target_dim_size  # Julia is 1-indexed
        pos = 1
        for i in 1:ndim
            if i == target_dim + 1
                continue
            end
            dims[i] = extra_dims[pos]
            pos += 1
        end
        return dims
    end

    # Helper function to compare arrays with relative error
    function compare_with_relative_error(a, b, tolerance)
        diff = abs.(a - b)
        ref = abs.(a)
        max_diff = maximum(diff)
        max_ref = maximum(ref)
        return max_diff <= tolerance * max_ref
    end

    @testset "Multi-dimensional Tensor Operations" begin
        status = Ref{Int32}(0)

        # Create basic setup
        k = LibSparseIR.spir_logistic_kernel_new(β * ωmax, status)
        @test status[] == 0
        sve = LibSparseIR.spir_sve_result_new(k, ε, status)
        @test status[] == 0
        basis = LibSparseIR.spir_basis_new(LibSparseIR.SPIR_STATISTICS_BOSONIC, β, ωmax, k, sve, status)
        @test status[] == 0

        basis_size_ref = Ref{Cint}(0)
        status_val = LibSparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test status_val == 0
        basis_size = basis_size_ref[]

        # Test tau sampling points
        num_tau_points_ref = Ref{Cint}(0)
        status_val = LibSparseIR.spir_basis_get_n_default_taus(basis, num_tau_points_ref)
        @test status_val == 0
        num_tau_points = num_tau_points_ref[]

        tau_points = zeros(Float64, num_tau_points)
        status_val = LibSparseIR.spir_basis_get_default_taus(basis, pointer(tau_points))
        @test status_val == 0

        tau_sampling = LibSparseIR.spir_tau_sampling_new(basis, num_tau_points, tau_points, status)
        @test status[] == 0

        # Test multi-dimensional tensor operations
        extra_dims = [2, 3, 4]
        ndim = 4

        for target_dim in 0:(ndim-1)
            println("Testing target_dim = $target_dim")

            # Get dimensions for different shapes
            dims_basis = get_dims(basis_size, extra_dims, target_dim, ndim)
            dims_tau = get_dims(num_tau_points, extra_dims, target_dim, ndim)

            # Create random coefficients tensor
            total_size_basis = prod(dims_basis)
            coeffs = rand(Float64, total_size_basis) .- 0.5

            # Test sampling evaluation: coeffs -> tau values
            total_size_tau = prod(dims_tau)
            result_tau = zeros(Float64, total_size_tau)

            status_val = LibSparseIR.spir_sampling_eval_dd(
                tau_sampling,
                LibSparseIR.SPIR_ORDER_COLUMN_MAJOR,
                ndim,
                dims_basis,
                target_dim,
                coeffs,
                result_tau
            )
            @test status_val == 0

            # Test sampling fitting: tau values -> coeffs
            coeffs_reconst = zeros(Float64, total_size_basis)

            status_val = LibSparseIR.spir_sampling_fit_dd(
                tau_sampling,
                LibSparseIR.SPIR_ORDER_COLUMN_MAJOR,
                ndim,
                dims_tau,
                target_dim,
                result_tau,
                coeffs_reconst
            )
            @test status_val == 0

            # Check round-trip accuracy
            @test compare_with_relative_error(coeffs, coeffs_reconst, tol)
        end

        LibSparseIR.spir_sampling_release(tau_sampling)
        LibSparseIR.spir_basis_release(basis)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(k)
    end

    @testset "Row-Major vs Column-Major Storage" begin
        status = Ref{Int32}(0)

        k = LibSparseIR.spir_logistic_kernel_new(β * ωmax, status)
        @test status[] == 0
        sve = LibSparseIR.spir_sve_result_new(k, ε, status)
        @test status[] == 0
        basis = LibSparseIR.spir_basis_new(LibSparseIR.SPIR_STATISTICS_BOSONIC, β, ωmax, k, sve, status)
        @test status[] == 0

        basis_size_ref = Ref{Cint}(0)
        status_val = LibSparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test status_val == 0
        basis_size = basis_size_ref[]

        num_tau_points_ref = Ref{Cint}(0)
        status_val = LibSparseIR.spir_basis_get_n_default_taus(basis, num_tau_points_ref)
        @test status_val == 0
        num_tau_points = num_tau_points_ref[]

        tau_points = zeros(Float64, num_tau_points)
        status_val = LibSparseIR.spir_basis_get_default_taus(basis, pointer(tau_points))
        @test status_val == 0

        tau_sampling = LibSparseIR.spir_tau_sampling_new(basis, num_tau_points, tau_points, status)
        @test status[] == 0

        # Test 2D case with different storage orders
        dims_2d = Int32[basis_size, 5]
        dims_tau_2d = Int32[num_tau_points, 5]
        coeffs_2d = rand(Float64, prod(dims_2d)) .- 0.5

        # Test Column-Major
        result_colmajor = zeros(Float64, prod(dims_tau_2d))
        status_val = LibSparseIR.spir_sampling_eval_dd(
            tau_sampling,
            LibSparseIR.SPIR_ORDER_COLUMN_MAJOR,
            2,
            dims_2d,
            0,  # target_dim = 0
            coeffs_2d,
            result_colmajor
        )
        @test status_val == 0

        # Test Row-Major
        result_rowmajor = zeros(Float64, prod(dims_tau_2d))
        status_val = LibSparseIR.spir_sampling_eval_dd(
            tau_sampling,
            LibSparseIR.SPIR_ORDER_ROW_MAJOR,
            2,
            dims_2d,
            0,  # target_dim = 0
            coeffs_2d,
            result_rowmajor
        )
        @test status_val == 0

        # Results should be different due to different memory layout interpretation
        @test !compare_with_relative_error(result_colmajor, result_rowmajor, 1e-15)

        # But round-trip should work for each
        coeffs_reconst_colmajor = zeros(Float64, prod(dims_2d))
        status_val = LibSparseIR.spir_sampling_fit_dd(
            tau_sampling,
            LibSparseIR.SPIR_ORDER_COLUMN_MAJOR,
            2,
            dims_tau_2d,
            0,
            result_colmajor,
            coeffs_reconst_colmajor
        )
        @test status_val == 0
        @test compare_with_relative_error(coeffs_2d, coeffs_reconst_colmajor, tol)

        coeffs_reconst_rowmajor = zeros(Float64, prod(dims_2d))
        status_val = LibSparseIR.spir_sampling_fit_dd(
            tau_sampling,
            LibSparseIR.SPIR_ORDER_ROW_MAJOR,
            2,
            dims_tau_2d,
            0,
            result_rowmajor,
            coeffs_reconst_rowmajor
        )
        @test status_val == 0
        @test compare_with_relative_error(coeffs_2d, coeffs_reconst_rowmajor, tol)

        LibSparseIR.spir_sampling_release(tau_sampling)
        LibSparseIR.spir_basis_release(basis)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(k)
    end

    @testset "Complex Tensor Operations" begin
        status = Ref{Int32}(0)

        k = LibSparseIR.spir_logistic_kernel_new(β * ωmax, status)
        @test status[] == 0
        sve = LibSparseIR.spir_sve_result_new(k, ε, status)
        @test status[] == 0
        basis = LibSparseIR.spir_basis_new(LibSparseIR.SPIR_STATISTICS_FERMIONIC, β, ωmax, k, sve, status)
        @test status[] == 0

        basis_size_ref = Ref{Cint}(0)
        status_val = LibSparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test status_val == 0
        basis_size = basis_size_ref[]

        # Get Matsubara sampling points
        positive_only = false
        num_matsubara_points_ref = Ref{Cint}(0)
        status_val = LibSparseIR.spir_basis_get_nmatuss(basis, positive_only, num_matsubara_points_ref)
        @test status_val == 0
        num_matsubara_points = num_matsubara_points_ref[]

        matsubara_indices = zeros(Int64, num_matsubara_points)
        status_val = LibSparseIR.spir_basis_get_matsus(basis, positive_only, pointer(matsubara_indices))
        @test status_val == 0

        matsu_sampling = LibSparseIR.spir_matsu_sampling_new(basis, positive_only, num_matsubara_points, matsubara_indices, status)
        @test status[] == 0

        # Test complex coefficient operations
        dims_2d = Int32[basis_size, 3]
        dims_matsu_2d = Int32[num_matsubara_points, 3]

        # Create complex coefficients (real IR coefficients)
        coeffs_real = rand(Float64, prod(dims_2d)) .- 0.5

        # Evaluate at Matsubara frequencies (produces complex result)
        result_complex = zeros(ComplexF64, prod(dims_matsu_2d))
        status_val = LibSparseIR.spir_sampling_eval_dz(
            matsu_sampling,
            LibSparseIR.SPIR_ORDER_COLUMN_MAJOR,
            2,
            dims_2d,
            0,
            coeffs_real,
            reinterpret(ComplexF32, result_complex)
        )
        @test status_val == 0

        # Fit back to get complex IR coefficients
        coeffs_complex = zeros(ComplexF64, prod(dims_2d))
        status_val = LibSparseIR.spir_sampling_fit_zz(
            matsu_sampling,
            LibSparseIR.SPIR_ORDER_COLUMN_MAJOR,
            2,
            dims_matsu_2d,
            0,
            reinterpret(ComplexF32, result_complex),
            reinterpret(ComplexF32, coeffs_complex)
        )
        @test status_val == 0

        # Check that the real part matches original coefficients
        @test compare_with_relative_error(coeffs_real, real.(coeffs_complex), tol)
        # Imaginary part should be small (approximately zero)
        @test maximum(abs.(imag.(coeffs_complex))) < tol

        LibSparseIR.spir_sampling_release(matsu_sampling)
        LibSparseIR.spir_basis_release(basis)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(k)
    end
end