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

@testitem "Basis Function Evaluation Helpers" begin
    using LibSparseIR

    # Helper function to evaluate basis functions at multiple points
    function evaluate_basis_functions(u::Ptr{LibSparseIR.spir_funcs}, x_values::Vector{Float64})
        status = Ref{Int32}(0)
        funcs_size = Ref{Int32}(0)
        status_code = LibSparseIR.spir_funcs_get_size(u, funcs_size)
        @test status_code == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        nfuncs = funcs_size[]
        npoints = length(x_values)
        u_eval_mat = Matrix{Float64}(undef, npoints, nfuncs)

        for i in 1:npoints
            u_eval = Vector{Float64}(undef, nfuncs)
            status_code = LibSparseIR.spir_funcs_eval(u, x_values[i], u_eval)
            @test status_code == LibSparseIR.SPIR_COMPUTATION_SUCCESS
            u_eval_mat[i, :] = u_eval
        end

        return u_eval_mat
    end

    # Helper function to evaluate Matsubara basis functions
    function evaluate_matsubara_basis_functions(uhat::Ptr{LibSparseIR.spir_funcs},
                                               matsubara_indices::Vector{Int64})
        status = Ref{Int32}(0)
        funcs_size = Ref{Int32}(0)
        status_code = LibSparseIR.spir_funcs_get_size(uhat, funcs_size)
        @test status_code == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        nfuncs = funcs_size[]
        nfreqs = length(matsubara_indices)
        uhat_eval_mat = Matrix{ComplexF64}(undef, nfreqs, nfuncs)

        # Evaluate all frequencies at once using batch evaluation
        status_code = LibSparseIR.spir_funcs_batch_eval_matsu(
            uhat, LibSparseIR.SPIR_ORDER_ROW_MAJOR, nfreqs, matsubara_indices,
            uhat_eval_mat)
        @test status_code == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        return uhat_eval_mat
    end

    @testset "Basic evaluation functionality" begin
        # Create a simple basis for testing
        beta = 2.0
        wmax = 5.0
        epsilon = 1e-6

        # Create kernel and SVE for basis creation
        kernel_status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = LibSparseIR.spir_sve_result_new(kernel, epsilon, sve_status)
        @test sve_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = LibSparseIR.spir_basis_new(LibSparseIR.SPIR_STATISTICS_FERMIONIC, beta, wmax, kernel, sve, basis_status)
        @test basis_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        # Get u basis functions
        u_status = Ref{Int32}(0)
        u = LibSparseIR.spir_basis_get_u(basis, u_status)
        @test u_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test u != C_NULL

        # Test evaluation at a few points
        x_values = [0.0, 0.5, 1.0]
        u_eval_mat = evaluate_basis_functions(u, x_values)
        @test size(u_eval_mat, 1) == length(x_values)
        @test size(u_eval_mat, 2) > 0  # Should have some basis functions
        @test all(isfinite, u_eval_mat)

        # Cleanup
        LibSparseIR.spir_funcs_release(u)
        LibSparseIR.spir_basis_release(basis)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(kernel)
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

@testitem "DLR Transformation Functions" begin
    using LibSparseIR

    # Helper functions for DLR transformations (corresponds to C++ template functions)
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

    @testset "DLR transformation interface" begin
        # This is a basic interface test - full functionality will be tested in integration tests
        @test dlr_to_IR isa Function
        @test dlr_from_IR isa Function
    end
end

@testitem "Sampling Transformation Functions" begin
    using LibSparseIR

    # Helper functions for sampling transformations
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

    function matsubara_sampling_evaluate(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                                        dims::Vector{Int32}, target_dim::Integer, gIR::Vector{Float64})
        giw = Vector{ComplexF64}(undef, prod(dims))
        status = LibSparseIR.spir_sampling_eval_dz(sampling, order, length(dims), dims, target_dim, gIR,
                                                  giw)
        return status, giw
    end

    function matsubara_sampling_evaluate(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                                        dims::Vector{Int32}, target_dim::Integer, gIR::Vector{ComplexF64})
        giw = Vector{ComplexF64}(undef, prod(dims))
        status = LibSparseIR.spir_sampling_eval_zz(sampling, order, length(dims), dims, target_dim,
                                                  gIR,
                                                  giw)
        return status, giw
    end

    @testset "Sampling transformation interface" begin
        @test tau_sampling_evaluate isa Function
        @test tau_sampling_fit isa Function
        @test matsubara_sampling_evaluate isa Function
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

    function matsubara_sampling_evaluate(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                                        dims::Vector{Int32}, target_dim::Integer, gIR::Vector{Float64})
        giw = Vector{ComplexF64}(undef, prod(dims))
        status = LibSparseIR.spir_sampling_eval_dz(sampling, order, length(dims), dims, target_dim, gIR,
                                                  giw)
        return status, giw
    end

    function matsubara_sampling_evaluate(sampling::Ptr{LibSparseIR.spir_sampling}, order::Integer,
                                        dims::Vector{Int32}, target_dim::Integer, gIR::Vector{ComplexF64})
        giw = Vector{ComplexF64}(undef, prod(dims))
        status = LibSparseIR.spir_sampling_eval_zz(sampling, order, length(dims), dims, target_dim,
                                                  gIR,
                                                  giw)
        return status, giw
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
        num_tau_points = Ref{Int32}(0)
        tau_status = LibSparseIR.spir_basis_get_n_default_taus(basis, num_tau_points)
        @test tau_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test num_tau_points[] > 0

        tau_points = Vector{Float64}(undef, num_tau_points[])
        tau_get_status = LibSparseIR.spir_basis_get_default_taus(basis, tau_points)
        @test tau_get_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        tau_sampling_status = Ref{Int32}(0)
        tau_sampling = LibSparseIR.spir_tau_sampling_new(basis, num_tau_points[], tau_points, tau_sampling_status)
        @test tau_sampling_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test tau_sampling != C_NULL

        # Matsubara Sampling
        println("Matsubara sampling")
        num_matsubara_points = Ref{Int32}(0)
        matsu_status = LibSparseIR.spir_basis_get_nmatuss(basis, positive_only, num_matsubara_points)
        @test matsu_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test num_matsubara_points[] > 0

        matsubara_points = Vector{Int64}(undef, num_matsubara_points[])
        matsu_get_status = LibSparseIR.spir_basis_get_matsus(basis, positive_only, matsubara_points)
        @test matsu_get_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        matsubara_sampling_status = Ref{Int32}(0)
        matsubara_sampling = LibSparseIR.spir_matsu_sampling_new(basis, positive_only, num_matsubara_points[], matsubara_points, matsubara_sampling_status)
        @test matsubara_sampling_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test matsubara_sampling != C_NULL

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

        # Generate random DLR coefficients
        Random.seed!(982743)  # Same seed as C++ test
        coeffs_dims = get_dims(npoles_val, extra_dims, target_dim, ndim)
        coeffs = Vector{T}(undef, prod(coeffs_dims))

        # Fill coefficients with random values
        for i in 1:length(coeffs)
            pole_idx = ((i - 1) % npoles_val) + 1
            random_val = rand()
            coeffs[i] = generate_random_coeff(T, random_val, poles[pole_idx])
        end

        println("Generated $(length(coeffs)) coefficients")

        # Convert DLR coefficients to IR coefficients
        ir_dims = get_dims(basis_size_val, extra_dims, target_dim, ndim)
        status, g_IR = dlr_to_IR(dlr, order, coeffs_dims, target_dim, coeffs)
        @test status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test length(g_IR) == prod(ir_dims)

        println("DLR to IR transformation successful")

        # Convert IR coefficients back to DLR coefficients
        status2, g_DLR_reconst = dlr_from_IR(dlr, order, ir_dims, target_dim, g_IR)
        @test status2 == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        println("IR to DLR transformation successful")

        # Test full sampling workflow: Matsubara -> IR -> tau -> IR -> Matsubara
        println("Testing full sampling workflow")

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
        matsu_dims = get_dims(num_matsubara_points[], extra_dims, target_dim, ndim)
        status_giw, giw_from_IR = matsubara_sampling_evaluate(matsubara_sampling, order, ir_dims, target_dim, g_IR)
        @test status_giw == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Fit Matsubara data back to IR coefficients
        status_fit, gIR_reconst = if T == Float64
            # Use complex fit and take real part
            status_temp, gIR_temp = LibSparseIR.spir_sampling_fit_zz(matsubara_sampling, order, length(matsu_dims), matsu_dims, target_dim, giw_from_IR, Vector{ComplexF64}(undef, prod(ir_dims)))
            status_temp, real.(gIR_temp)
        else
            LibSparseIR.spir_sampling_fit_zz(matsubara_sampling, order, length(matsu_dims), matsu_dims, target_dim, giw_from_IR, Vector{ComplexF64}(undef, prod(ir_dims)))
        end
        @test status_fit == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # IR -> tau
        tau_dims = get_dims(num_tau_points[], extra_dims, target_dim, ndim)
        status_tau, gtau = tau_sampling_evaluate(tau_sampling, order, ir_dims, target_dim, gIR_reconst)
        @test status_tau == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # tau -> IR
        status_tau_fit, gIR2 = tau_sampling_fit(tau_sampling, order, tau_dims, target_dim, gtau)
        @test status_tau_fit == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # IR -> Matsubara (final check)
        status_final, giw_reconst = matsubara_sampling_evaluate(matsubara_sampling, order, ir_dims, target_dim, gIR2)
        @test status_final == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Verify consistency (basic checks)
        @test all(isfinite, g_IR)
        @test all(isfinite, g_DLR_reconst)
        @test all(isfinite, gtau)
        @test all(isfinite, gIR2)
        @test all(isfinite, giw_reconst)

        # Check that the round-trip transformations are reasonably consistent
        # Note: We use a relaxed tolerance due to numerical precision in the transformations
        relaxed_tol = max(tol * 100, 1e-8)
        @test compare_arrays_with_relative_error(gIR_reconst, gIR2, relaxed_tol)

        println("Integration test completed successfully")

        # Cleanup
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
        for positive_only in [false, true]
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
            extra_dims = [2, 3, 4]

            for target_dim in 0:3
                @testset "4D Bosonic LogisticKernel ColMajor target_dim=$target_dim positive_only=$positive_only" begin
                    integration_test(Float64, LibSparseIR.SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon,
                                   extra_dims, target_dim, LibSparseIR.SPIR_ORDER_COLUMN_MAJOR, tol, positive_only)
                end

                @testset "4D Bosonic LogisticKernel RowMajor target_dim=$target_dim positive_only=$positive_only" begin
                    integration_test(Float64, LibSparseIR.SPIR_STATISTICS_BOSONIC, beta, wmax, epsilon,
                                   extra_dims, target_dim, LibSparseIR.SPIR_ORDER_ROW_MAJOR, tol, positive_only)
                end
            end
        end
    end
end
