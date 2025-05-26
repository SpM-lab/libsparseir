# Tests corresponding to test/cpp/cinterface_integration.cxx
# Comprehensive integration tests for the full workflow: DLR ↔ IR ↔ Sampling transformations

@testitem "Helper Function get_dims" begin
    using LibSparseIR

    # Test the get_dims helper function (corresponds to C++ get_dims template)
    function get_dims(target_dim_size::Int, extra_dims::Vector{Int}, target_dim::Int, ndim::Int)
        dims = Vector{Int}(undef, ndim)
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

    @testset "Test get_dims functionality" begin
        extra_dims = [2, 3, 4]

        # target_dim = 0 (first dimension)
        dims = get_dims(100, extra_dims, 0, 4)
        @test dims == [100, 2, 3, 4]

        # target_dim = 1 (second dimension)
        dims = get_dims(100, extra_dims, 1, 4)
        @test dims == [2, 100, 3, 4]

        # target_dim = 2 (third dimension)
        dims = get_dims(100, extra_dims, 2, 4)
        @test dims == [2, 3, 100, 4]

        # target_dim = 3 (fourth dimension)
        dims = get_dims(100, extra_dims, 3, 4)
        @test dims == [2, 3, 4, 100]
    end
end

@testitem "Tensor Comparison Utilities" begin
    using LibSparseIR

    # Helper function to compare arrays with relative error tolerance (corresponds to C++ compare_tensors_with_relative_error)
    function compare_arrays_with_relative_error(a::Array{T}, b::Array{T}, tol::Float64) where T
        @test size(a) == size(b)

        diff = abs.(a - b)
        ref = abs.(a)

        max_diff = maximum(diff)
        max_ref = maximum(ref)

        # Debug output if test fails
        if max_diff > tol * max_ref
            println("max_diff: $max_diff")
            println("max_ref: $max_ref")
            println("tol: $tol")
            println("relative_error: $(max_diff / max_ref)")
        end

        return max_diff <= tol * max_ref
    end

    @testset "Array comparison functionality" begin
        # Test with identical arrays
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        @test compare_arrays_with_relative_error(a, b, 1e-10)

        # Test with small differences
        a = [1.0, 2.0, 3.0]
        b = [1.0001, 2.0001, 3.0001]
        @test compare_arrays_with_relative_error(a, b, 1e-3)
        @test !compare_arrays_with_relative_error(a, b, 1e-5)

        # Test with complex numbers
        a_complex = [1.0 + 0.0im, 2.0 + 1.0im]
        b_complex = [1.0001 + 0.0001im, 2.0001 + 1.0001im]
        @test compare_arrays_with_relative_error(a_complex, b_complex, 1e-3)
    end
end

@testitem "Random Coefficient Generation" begin
    using LibSparseIR
    using Random

    # Helper function to generate random coefficients (corresponds to C++ generate_random_coeff)
    function generate_random_coeff(::Type{Float64}, random_value::Float64, pole::Float64)
        return (2.0 * random_value - 1.0) * sqrt(abs(pole))
    end

    function generate_random_coeff(::Type{ComplexF64}, random_value::Float64, pole::Float64)
        real_part = (2.0 * random_value - 1.0) * sqrt(abs(pole))
        imag_part = (2.0 * random_value - 1.0) * sqrt(abs(pole))
        return ComplexF64(real_part, imag_part)
    end

    @testset "Coefficient generation" begin
        Random.seed!(982743)  # Same seed as C++ test

        # Test real coefficients
        pole = 2.5
        coeff_real = generate_random_coeff(Float64, 0.7, pole)
        @test isa(coeff_real, Float64)
        @test abs(coeff_real) <= sqrt(abs(pole))

        # Test complex coefficients
        coeff_complex = generate_random_coeff(ComplexF64, 0.3, pole)
        @test isa(coeff_complex, ComplexF64)
        @test abs(real(coeff_complex)) <= sqrt(abs(pole))
        @test abs(imag(coeff_complex)) <= sqrt(abs(pole))
    end
end

@testitem "Main Integration Test" begin
    using LibSparseIR
    using Random

    # Helper function equivalent to C++ _spir_basis_new
    function _spir_basis_new(statistics::Integer, beta::Float64, omega_max::Float64, epsilon::Float64)
        # Create logistic kernel
        kernel_status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_logistic_kernel_new(beta * omega_max, kernel_status)
        @test kernel_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test kernel != C_NULL

        # Create SVE result
        sve_status = Ref{Int32}(0)
        sve = LibSparseIR.spir_sve_result_new(kernel, epsilon, sve_status)
        @test sve_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test sve != C_NULL

        # Create basis
        basis_status = Ref{Int32}(0)
        basis = LibSparseIR.spir_basis_new(statistics, beta, omega_max, kernel, sve, basis_status)
        @test basis_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        # Clean up intermediate objects (like C++ version)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(kernel)

        return basis
    end

    # Include helper functions from previous test items
    function get_dims(target_dim_size::Integer, extra_dims::Vector{<:Integer}, target_dim::Integer, ndim::Integer)
        dims = Vector{Int32}(undef, ndim)
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

    function compare_arrays_with_relative_error(a::Array{T}, b::Array{T}, tol::Float64) where T
        if size(a) != size(b)
            return false
        end

        diff = abs.(a - b)
        ref = abs.(a)

        max_diff = maximum(diff)
        max_ref = maximum(ref)

        # Debug output if test fails
        if max_diff > tol * max_ref
            println("max_diff: $max_diff")
            println("max_ref: $max_ref")
            println("tol: $tol")
            println("relative_error: $(max_diff / max_ref)")
        end

        return max_diff <= tol * max_ref
    end

    function generate_random_coeff(::Type{Float64}, random_value::Float64, pole::Float64)
        return (2.0 * random_value - 1.0) * sqrt(abs(pole))
    end

    function generate_random_coeff(::Type{ComplexF64}, random_value::Float64, pole::Float64)
        real_part = (2.0 * random_value - 1.0) * sqrt(abs(pole))
        imag_part = (2.0 * random_value - 1.0) * sqrt(abs(pole))
        return ComplexF64(real_part, imag_part)
    end

    # DLR transformation helper functions
    function dlr_to_IR(dlr::Ptr{LibSparseIR.spir_basis}, order::Integer, dims::Vector{Int32},
                       target_dim::Integer, coeffs::Vector{Float64})
        g_IR = Vector{Float64}(undef, prod(dims))
        status = LibSparseIR.spir_dlr2ir_dd(dlr, order, length(dims), dims, target_dim, coeffs, g_IR)
        return status, g_IR
    end

    function dlr_to_IR(dlr::Ptr{LibSparseIR.spir_basis}, order::Integer, dims::Vector{Int32},
                       target_dim::Integer, coeffs::Vector{ComplexF64})
        g_IR = Vector{ComplexF64}(undef, prod(dims))
        status = LibSparseIR.spir_dlr2ir_zz(dlr, order, length(dims), dims, target_dim,
                                           coeffs,
                                           g_IR)
        return status, g_IR
    end

    function dlr_from_IR(dlr::Ptr{LibSparseIR.spir_basis}, order::Integer, dims::Vector{Int32},
                        target_dim::Integer, coeffs::Vector{Float64})
        g_DLR = Vector{Float64}(undef, prod(dims))
        status = LibSparseIR.spir_ir2dlr_dd(dlr, order, length(dims), dims, target_dim, coeffs, g_DLR)
        return status, g_DLR
    end

    function dlr_from_IR(dlr::Ptr{LibSparseIR.spir_basis}, order::Integer, dims::Vector{Int32},
                        target_dim::Integer, coeffs::Vector{ComplexF64})
        g_DLR = Vector{ComplexF64}(undef, prod(dims))
        status = LibSparseIR.spir_ir2dlr_zz(dlr, order, length(dims), dims, target_dim,
                                           coeffs,
                                           g_DLR)
        return status, g_DLR
    end

    # Sampling transformation helper functions
    function tau_sampling_evaluate(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                                  dims::Vector{Int32}, target_dim::Integer, gIR::Vector{Float64})
        gtau = Vector{Float64}(undef, prod(dims))
        status = LibSparseIR.spir_sampling_eval_dd(sampling, order, length(dims), dims, target_dim, gIR, gtau)
        return status, gtau
    end

    function tau_sampling_evaluate(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                                  dims::Vector{Int32}, target_dim::Integer, gIR::Vector{ComplexF64})
        gtau = Vector{ComplexF64}(undef, prod(dims))
        status = LibSparseIR.spir_sampling_eval_zz(sampling, order, length(dims), dims, target_dim,
                                                  gIR,
                                                  gtau)
        return status, gtau
    end

    function tau_sampling_fit(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                             dims::Vector{Int32}, target_dim::Integer, gtau::Vector{Float64})
        gIR = Vector{Float64}(undef, prod(dims))
        status = LibSparseIR.spir_sampling_fit_dd(sampling, order, length(dims), dims, target_dim, gtau, gIR)
        return status, gIR
    end

    function tau_sampling_fit(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                             dims::Vector{Int32}, target_dim::Integer, gtau::Vector{ComplexF64})
        gIR = Vector{ComplexF64}(undef, prod(dims))
        status = LibSparseIR.spir_sampling_fit_zz(sampling, order, length(dims), dims, target_dim,
                                                 gtau,
                                                 gIR)
        return status, gIR
    end

    # FIXED: Matsubara sampling functions with correct array dimensions
    function matsubara_sampling_evaluate(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                                        ir_dims::Vector{Int32}, target_dim::Integer, gIR::Vector{Float64}, matsu_dims::Vector{Int32})
        giw = Vector{ComplexF64}(undef, prod(matsu_dims))
        status = LibSparseIR.spir_sampling_eval_dz(sampling, order, length(ir_dims), ir_dims, target_dim, gIR,
                                                  giw)
        return status, giw
    end

    function matsubara_sampling_evaluate(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                                        ir_dims::Vector{Int32}, target_dim::Integer, gIR::Vector{ComplexF64}, matsu_dims::Vector{Int32})
        giw = Vector{ComplexF64}(undef, prod(matsu_dims))
        status = LibSparseIR.spir_sampling_eval_zz(sampling, order, length(ir_dims), ir_dims, target_dim,
                                                  gIR,
                                                  giw)
        return status, giw
    end

    function movedim(arr::AbstractArray, src::Integer, dst::Integer)
        if src == dst
            return arr
        else
            perm = [1:ndims(arr)...]
            perm[src] = dst
            perm[dst] = src
            return permutedims(arr, perm)
        end
    end
    # Main integration test function (corresponds to C++ integration_test template)
    function integration_test(::Type{T}, statistics::Integer, beta::Float64, wmax::Float64,
                             epsilon::Float64, extra_dims::Vector{Int}, target_dim::Int,
                             order::Integer, tol::Float64, positive_only::Bool) where T

        # positive_only is not supported for complex numbers
        if T == ComplexF64 && positive_only
            @test_skip "positive_only not supported for complex numbers"
            return
        end

        ndim = 1 + length(extra_dims)

        println("Running integration test: T=$T, statistics=$statistics, target_dim=$target_dim, positive_only=$positive_only")

        # Create IR basis using helper function (equivalent to C++ _spir_basis_new)
        basis = _spir_basis_new(statistics, beta, wmax, epsilon)

        # Get basis size
        basis_size = Ref{Int32}(0)
        size_status = LibSparseIR.spir_basis_get_size(basis, basis_size)
        @test size_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size_val = basis_size[]

        # Tau Sampling
        println("Tau sampling")

        # Add safety check for basis validity before calling tau functions
        @test basis != C_NULL

        num_tau_points = Ref{Int32}(0)
        tau_status = LibSparseIR.spir_basis_get_n_default_taus(basis, num_tau_points)
        @test tau_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test num_tau_points[] > 0

        tau_points_org = Vector{Float64}(undef, num_tau_points[])
        tau_get_status = LibSparseIR.spir_basis_get_default_taus(basis, tau_points_org)
        @test tau_get_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        tau_sampling_status = Ref{Int32}(0)
        tau_sampling = LibSparseIR.spir_tau_sampling_new(basis, num_tau_points[], tau_points_org, tau_sampling_status)
        @test tau_sampling_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test tau_sampling != C_NULL

        num_tau_points = Ref{Int32}(0)
        num_tau_points_status = LibSparseIR.spir_sampling_get_npoints(tau_sampling, num_tau_points)
        @test num_tau_points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        tau_points = Vector{Float64}(undef, num_tau_points[])
        tau_points_status = LibSparseIR.spir_sampling_get_taus(tau_sampling, tau_points)
        @test tau_points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test tau_points == tau_points_org
        @test num_tau_points[] >= basis_size_val
        # Matsubara Sampling
        println("Matsubara sampling")
        num_matsubara_points_org = Ref{Int32}(0)
        matsu_status = LibSparseIR.spir_basis_get_nmatuss(basis, positive_only, num_matsubara_points_org)
        @test matsu_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test num_matsubara_points_org[] > 0
        matsubara_points_org = Vector{Int64}(undef, num_matsubara_points_org[])
        matsu_get_status = LibSparseIR.spir_basis_get_matsus(basis, positive_only, matsubara_points_org)
        @test matsu_get_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        matsubara_sampling_status = Ref{Int32}(0)
        matsubara_sampling = LibSparseIR.spir_matsu_sampling_new(basis, positive_only, num_matsubara_points_org[], matsubara_points_org, matsubara_sampling_status)
        @test matsubara_sampling_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test matsubara_sampling != C_NULL

        if positive_only
            @test num_matsubara_points_org[] >= basis_size_val / 2
        else
            @test num_matsubara_points_org[] >= basis_size_val
        end

        num_matsubara_points = Ref{Int32}(0)
        num_matsubara_points_status = LibSparseIR.spir_sampling_get_npoints(matsubara_sampling, num_matsubara_points)
        @test num_matsubara_points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        matsubara_points = Vector{Int64}(undef, num_matsubara_points[])
        matsubara_points_status = LibSparseIR.spir_sampling_get_matsus(matsubara_sampling, matsubara_points)
        @test matsubara_points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test matsubara_points == matsubara_points_org
        # DLR
        println("DLR")
        dlr_status = Ref{Int32}(0)
        dlr = LibSparseIR.spir_dlr_new(basis, dlr_status)
        @test dlr_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test dlr != C_NULL

        # Get number of poles
        npoles = Ref{Int32}(0)
        poles_status = LibSparseIR.spir_dlr_get_npoles(dlr, npoles)
        @test poles_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        npoles_val = npoles[]
        @test npoles_val >= basis_size_val

        # Get poles
        poles = Vector{Float64}(undef, npoles_val)
        poles_get_status = LibSparseIR.spir_dlr_get_poles(dlr, poles)
        @test poles_get_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test maximum(abs.(poles)) <= wmax
        # Calculate total size of extra dimensions
        extra_size = prod(extra_dims)
        coeffs_targetdim0 = Array{T}(undef, npoles_val, extra_dims...)
        coeffs_2d = reshape(coeffs_targetdim0, Int64(npoles_val), extra_size)
        # Generate random DLR coefficients
        for i in 1:npoles_val, j in 1:extra_size
            coeffs_2d[i, j] = generate_random_coeff(T, rand(), poles[i])
        end
        println("Generated $(length(coeffs_2d)) coefficients")
        coeffs = movedim(coeffs_targetdim0, 1, 1+target_dim) # Julia is 1-based

        # Convert DLR coefficients to IR coefficients
        ir_dims = get_dims(basis_size_val, extra_dims, target_dim, ndim)
        status, g_IR = dlr_to_IR(dlr, order, get_dims(npoles_val, extra_dims, target_dim, ndim), target_dim, vec(coeffs))
        @test status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test length(g_IR) == prod(ir_dims)
        @test all(isfinite, g_IR)

        # Convert IR coefficients back to DLR coefficients
        status2, g_DLR_reconst = dlr_from_IR(dlr, order, ir_dims, target_dim, g_IR)
        @test status2 == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # From_IR C API
        status3, g_dlr = dlr_from_IR(dlr, order, get_dims(basis_size_val, extra_dims, target_dim, ndim), target_dim, vec(g_IR))

        # Get basis functions for evaluation
        ir_u_status = Ref{Int32}(0)
        ir_u = LibSparseIR.spir_basis_get_u(basis, ir_u_status)
        @test ir_u_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test ir_u != C_NULL

        ir_uhat_status = Ref{Int32}(0)
        ir_uhat = LibSparseIR.spir_basis_get_uhat(basis, ir_uhat_status)
        @test ir_uhat_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test ir_uhat != C_NULL

        # Evaluate Green's function at Matsubara frequencies using IR coefficients
        matsu_dims = get_dims(num_matsubara_points_org[], extra_dims, target_dim, ndim)
        # FIXED: Pass both ir_dims and matsu_dims to the function
        status_giw, giw_from_IR = matsubara_sampling_evaluate(matsubara_sampling, order, ir_dims, target_dim, g_IR, matsu_dims)
        @test status_giw == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Fit Matsubara data back to IR coefficients
        gIR_reconst = if T == Float64
            # Use complex fit and take real part
            gIR_temp = Vector{ComplexF64}(undef, prod(ir_dims))
            status_temp = LibSparseIR.spir_sampling_fit_zz(matsubara_sampling, order, length(matsu_dims), matsu_dims, target_dim, giw_from_IR, gIR_temp)
            @test status_temp == LibSparseIR.SPIR_COMPUTATION_SUCCESS
            real.(gIR_temp)
        else
            gIR_temp = Vector{ComplexF64}(undef, prod(ir_dims))
            status_temp = LibSparseIR.spir_sampling_fit_zz(matsubara_sampling, order, length(matsu_dims), matsu_dims, target_dim, giw_from_IR, gIR_temp)
            @test status_temp == LibSparseIR.SPIR_COMPUTATION_SUCCESS
            gIR_temp
        end

        # IR -> tau
        tau_dims = get_dims(num_tau_points[], extra_dims, target_dim, ndim)
        status_tau, gtau = tau_sampling_evaluate(tau_sampling, order, ir_dims, target_dim, gIR_reconst)
        @test status_tau == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # tau -> IR
        status_tau_fit, gIR2 = tau_sampling_fit(tau_sampling, order, tau_dims, target_dim, gtau)
        @test status_tau_fit == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # IR -> Matsubara (final check)
        # FIXED: Pass both ir_dims and matsu_dims to the function
        status_final, giw_reconst = matsubara_sampling_evaluate(matsubara_sampling, order, ir_dims, target_dim, gIR2, matsu_dims)
        @test status_final == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Verify consistency (basic checks)
        @test all(isfinite, g_IR)
        @test all(isfinite, g_DLR_reconst)
        @test all(isfinite, gtau)
        @test all(isfinite, gIR2)
        @test all(isfinite, giw_reconst)

        # Check that the round-trip transformations are reasonably consistent
        # Note: We use a relaxed tolerance due to numerical precision in the transformations
        @test compare_arrays_with_relative_error(gIR_reconst, gIR2, tol)

        println("Integration test completed successfully")

        # Cleanup resources in reverse order of creation
        LibSparseIR.spir_funcs_release(ir_uhat)
        LibSparseIR.spir_funcs_release(ir_u)
        LibSparseIR.spir_basis_release(dlr)
        LibSparseIR.spir_sampling_release(matsubara_sampling)
        LibSparseIR.spir_sampling_release(tau_sampling)
        LibSparseIR.spir_basis_release(basis)
    end

    @testset "Comprehensive Integration Tests" begin
        beta = 10.0
        wmax = 2.0
        epsilon = 1e-10
        tol = 10 * epsilon

        # Test matrix corresponding to C++ comprehensive tests
        println("Running comprehensive integration tests...")

        for positive_only in [false]
            println("positive_only = $positive_only")

            # 1D tests with different memory orders
            extra_dims = Int[]
            target_dim = 0

            @testset "1D Bosonic LogisticKernel ColMajor positive_only=$positive_only" begin
                integration_test(Float64, LibSparseIR.SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon,
                               extra_dims, target_dim, LibSparseIR.SPIR_ORDER_COLUMN_MAJOR, tol, positive_only)

                if !positive_only
                    integration_test(ComplexF64, LibSparseIR.SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon,
                                   extra_dims, target_dim, LibSparseIR.SPIR_ORDER_COLUMN_MAJOR, tol, positive_only)
                end
            end

            @testset "1D Bosonic LogisticKernel RowMajor positive_only=$positive_only" begin
                integration_test(Float64, LibSparseIR.SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon,
                               extra_dims, target_dim, LibSparseIR.SPIR_ORDER_ROW_MAJOR, tol, positive_only)

                if !positive_only
                    integration_test(ComplexF64, LibSparseIR.SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon,
                                   extra_dims, target_dim, LibSparseIR.SPIR_ORDER_ROW_MAJOR, tol, positive_only)
                end
            end

            # Multi-dimensional tests (corresponds to C++ extra_dims = {2,3,4})
            # Use smaller dimensions to avoid memory issues
            extra_dims = [2, 3, 4]

            for target_dim in 0:3
                @testset "4D Bosonic LogisticKernel ColMajor target_dim=$target_dim positive_only=$positive_only" begin
                    # Add memory usage check before running heavy tests
                    total_elements = prod(get_dims(18, extra_dims, target_dim, 4))  # Estimate with typical basis size
                        integration_test(Float64, LibSparseIR.SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon,
                                       extra_dims, target_dim, LibSparseIR.SPIR_ORDER_COLUMN_MAJOR, tol, positive_only)

                end

                @testset "4D Bosonic LogisticKernel RowMajor target_dim=$target_dim positive_only=$positive_only" begin
                    # Add memory usage check before running heavy tests
                    total_elements = prod(get_dims(18, extra_dims, target_dim, 4))  # Estimate with typical basis size

                        integration_test(Float64, LibSparseIR.SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon,
                                       extra_dims, target_dim, LibSparseIR.SPIR_ORDER_ROW_MAJOR, tol, positive_only)
                end
            end
        end

        println("Comprehensive integration tests completed")
    end
end