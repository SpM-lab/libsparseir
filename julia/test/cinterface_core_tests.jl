# Tests corresponding to test/cpp/cinterface_core.cxx
# Tests for kernel accuracy, basis constructors, and basis functions

@testitem "Kernel Accuracy Tests" begin
    using LibSparseIR

    # Test individual kernels (corresponds to cinterface_core.cxx TEST_CASE "Kernel Accuracy Tests")

    @testset "LogisticKernel(9)" begin
        Lambda = 9.0
        status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_logistic_kernel_new(Lambda, status)
        @test status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test kernel != C_NULL
        LibSparseIR.spir_kernel_release(kernel)
    end

    @testset "RegularizedBoseKernel(10)" begin
        Lambda = 10.0
        status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_reg_bose_kernel_new(Lambda, status)
        @test status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test kernel != C_NULL
        LibSparseIR.spir_kernel_release(kernel)
    end

    @testset "Kernel Domain" begin
        kernel_status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_logistic_kernel_new(9.0, kernel_status)
        @test kernel_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test kernel != C_NULL

        # Get domain bounds
        xmin = Ref{Float64}(0.0)
        xmax = Ref{Float64}(0.0)
        ymin = Ref{Float64}(0.0)
        ymax = Ref{Float64}(0.0)
        domain_status = LibSparseIR.spir_kernel_domain(kernel, xmin, xmax, ymin, ymax)
        @test domain_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # For LogisticKernel, we expect specific domain values
        @test xmin[] ≈ -1.0
        @test xmax[] ≈ 1.0
        @test ymin[] ≈ -1.0
        @test ymax[] ≈ 1.0

        LibSparseIR.spir_kernel_release(kernel)
    end
end

@testitem "FiniteTempBasis Constructor Tests" begin
    using LibSparseIR

    # Helper function equivalent to C++ _spir_basis_new
    function _spir_basis_new(statistics::Integer, beta::Float64, omega_max::Float64, epsilon::Float64)
        status = Ref{Int32}(0)

        # Create logistic kernel
        kernel_status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_logistic_kernel_new(beta * omega_max, kernel_status)
        if kernel_status[] != LibSparseIR.SPIR_COMPUTATION_SUCCESS || kernel == C_NULL
            return C_NULL, kernel_status[]
        end

        # Create SVE result
        sve_status = Ref{Int32}(0)
        sve = LibSparseIR.spir_sve_result_new(kernel, epsilon, sve_status)
        if sve_status[] != LibSparseIR.SPIR_COMPUTATION_SUCCESS || sve == C_NULL
            LibSparseIR.spir_kernel_release(kernel)
            return C_NULL, sve_status[]
        end

        # Create basis
        basis_status = Ref{Int32}(0)
        basis = LibSparseIR.spir_basis_new(statistics, beta, omega_max, kernel, sve, basis_status)
        if basis_status[] != LibSparseIR.SPIR_COMPUTATION_SUCCESS || basis == C_NULL
            LibSparseIR.spir_sve_result_release(sve)
            LibSparseIR.spir_kernel_release(kernel)
            return C_NULL, basis_status[]
        end

        # Clean up intermediate objects (like C++ version)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(kernel)

        return basis, LibSparseIR.SPIR_COMPUTATION_SUCCESS
    end

    # Test basis constructors (corresponds to cinterface_core.cxx TEST_CASE "FiniteTempBasis")

    function test_basis_constructor(statistics::Integer)
        beta = 2.0
        wmax = 5.0
        epsilon = 1e-6

        # Create basis using helper function (equivalent to C++ _spir_basis_new)
        basis, basis_status = _spir_basis_new(statistics, beta, wmax, epsilon)
        @test basis_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        # Check basis size
        basis_size = Ref{Int32}(0)
        size_result = LibSparseIR.spir_basis_get_size(basis, basis_size)
        @test size_result == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis_size[] > 0

        LibSparseIR.spir_basis_release(basis)
        return basis_size[]
    end

    function test_basis_constructor_with_sve(statistics::Integer, use_reg_bose=false)
        beta = 2.0
        wmax = 5.0
        Lambda = 10.0
        epsilon = 1e-6

        # Create kernel
        kernel_status = Ref{Int32}(0)
        kernel = if use_reg_bose
            LibSparseIR.spir_reg_bose_kernel_new(Lambda, kernel_status)
        else
            LibSparseIR.spir_logistic_kernel_new(Lambda, kernel_status)
        end
        @test kernel_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test kernel != C_NULL

        # Create SVE result
        sve_status = Ref{Int32}(0)
        sve_result = LibSparseIR.spir_sve_result_new(kernel, epsilon, sve_status)
        @test sve_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test sve_result != C_NULL

        # Create basis with SVE
        basis_status = Ref{Int32}(0)
        basis = LibSparseIR.spir_basis_new(statistics, beta, wmax, kernel, sve_result, basis_status)
        @test basis_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        # Check statistics
        stats = Ref{Int32}(0)
        stats_status = LibSparseIR.spir_basis_get_stats(basis, stats)
        @test stats_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test stats[] == statistics

        # Clean up
        LibSparseIR.spir_kernel_release(kernel)
        LibSparseIR.spir_sve_result_release(sve_result)
        LibSparseIR.spir_basis_release(basis)
    end

    @testset "FiniteTempBasis Constructor Fermionic" begin
        test_basis_constructor(LibSparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "FiniteTempBasis Constructor Bosonic" begin
        test_basis_constructor(LibSparseIR.SPIR_STATISTICS_BOSONIC)
    end

    @testset "FiniteTempBasis Constructor with SVE Fermionic/LogisticKernel" begin
        test_basis_constructor_with_sve(LibSparseIR.SPIR_STATISTICS_FERMIONIC, false)
    end

    @testset "FiniteTempBasis Constructor with SVE Bosonic/LogisticKernel" begin
        test_basis_constructor_with_sve(LibSparseIR.SPIR_STATISTICS_BOSONIC, false)
    end

    @testset "FiniteTempBasis Constructor with SVE Bosonic/RegularizedBoseKernel" begin
        test_basis_constructor_with_sve(LibSparseIR.SPIR_STATISTICS_BOSONIC, true)
    end
end

@testitem "FiniteTempBasis Basis Functions Tests" begin
    using LibSparseIR

    # Helper function equivalent to C++ _spir_basis_new
    function _spir_basis_new(statistics::Integer, beta::Float64, omega_max::Float64, epsilon::Float64)
        status = Ref{Int32}(0)

        # Create logistic kernel
        kernel_status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_logistic_kernel_new(beta * omega_max, kernel_status)
        if kernel_status[] != LibSparseIR.SPIR_COMPUTATION_SUCCESS || kernel == C_NULL
            return C_NULL, kernel_status[]
        end

        # Create SVE result
        sve_status = Ref{Int32}(0)
        sve = LibSparseIR.spir_sve_result_new(kernel, epsilon, sve_status)
        if sve_status[] != LibSparseIR.SPIR_COMPUTATION_SUCCESS || sve == C_NULL
            LibSparseIR.spir_kernel_release(kernel)
            return C_NULL, sve_status[]
        end

        # Create basis
        basis_status = Ref{Int32}(0)
        basis = LibSparseIR.spir_basis_new(statistics, beta, omega_max, kernel, sve, basis_status)
        if basis_status[] != LibSparseIR.SPIR_COMPUTATION_SUCCESS || basis == C_NULL
            LibSparseIR.spir_sve_result_release(sve)
            LibSparseIR.spir_kernel_release(kernel)
            return C_NULL, basis_status[]
        end

        # Clean up intermediate objects (like C++ version)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(kernel)

        return basis, LibSparseIR.SPIR_COMPUTATION_SUCCESS
    end

    # Test basis function evaluation (corresponds to cinterface_core.cxx TEST_CASE "FiniteTempBasis Basis Functions")

    function test_basis_functions(statistics::Integer)
        beta = 2.0
        wmax = 5.0
        epsilon = 1e-6

        # Create basis using helper function (equivalent to C++ _spir_basis_new)
        basis, basis_status = _spir_basis_new(statistics, beta, wmax, epsilon)
        @test basis_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        # Get basis size
        basis_size = Ref{Int32}(0)
        size_status = LibSparseIR.spir_basis_get_size(basis, basis_size)
        @test size_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        size = basis_size[]

        # Get u basis functions
        u_status = Ref{Int32}(0)
        u = LibSparseIR.spir_basis_get_u(basis, u_status)
        @test u_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test u != C_NULL

        # Get uhat basis functions
        uhat_status = Ref{Int32}(0)
        uhat = LibSparseIR.spir_basis_get_uhat(basis, uhat_status)
        @test uhat_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test uhat != C_NULL

        # Get v basis functions
        v_status = Ref{Int32}(0)
        v = LibSparseIR.spir_basis_get_v(basis, v_status)
        @test v_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test v != C_NULL

        # Test single point evaluation for u basis
        x = 0.5  # Test point for u basis (imaginary time)
        out = Vector{Float64}(undef, size)
        eval_status = LibSparseIR.spir_funcs_eval(u, x, out)
        @test eval_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Check that we got reasonable values
        @test all(isfinite, out)

        # Test single point evaluation for v basis
        y = 0.5 * wmax  # Test point for v basis (real frequency)
        eval_status = LibSparseIR.spir_funcs_eval(v, y, out)
        @test eval_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test all(isfinite, out)

        # Test batch evaluation
        num_points = 5
        xs = [0.2 * (i) for i in 1:num_points]  # Points at 0.2, 0.4, 0.6, 0.8, 1.0
        batch_out = Vector{Float64}(undef, num_points * size)

        # Test row-major order for u basis
        batch_status = LibSparseIR.spir_funcs_batch_eval(
            u, LibSparseIR.SPIR_ORDER_ROW_MAJOR, num_points, xs, batch_out)
        @test batch_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test all(isfinite, batch_out)

        # Test column-major order for u basis
        batch_status = LibSparseIR.spir_funcs_batch_eval(
            u, LibSparseIR.SPIR_ORDER_COLUMN_MAJOR, num_points, xs, batch_out)
        @test batch_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test all(isfinite, batch_out)

        # Test row-major order for v basis
        batch_status = LibSparseIR.spir_funcs_batch_eval(
            v, LibSparseIR.SPIR_ORDER_ROW_MAJOR, num_points, xs, batch_out)
        @test batch_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test all(isfinite, batch_out)

        # Test column-major order for v basis
        batch_status = LibSparseIR.spir_funcs_batch_eval(
            v, LibSparseIR.SPIR_ORDER_COLUMN_MAJOR, num_points, xs, batch_out)
        @test batch_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test all(isfinite, batch_out)

        # Test error cases (corresponds to C++ error case testing)
        # Test with null function pointer
        eval_status = LibSparseIR.spir_funcs_eval(C_NULL, x, out)
        @test eval_status != LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Test with null output array - create a null pointer for testing
        # Note: In Julia, we can't easily pass null for the output array as it would be unsafe
        # This corresponds to spir_funcs_eval(u, x, nullptr) in C++
        # We skip this test for safety reasons in Julia

        # Test batch evaluation error cases
        batch_status = LibSparseIR.spir_funcs_batch_eval(
            C_NULL, LibSparseIR.SPIR_ORDER_ROW_MAJOR, num_points, xs, batch_out)
        @test batch_status != LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Clean up
        LibSparseIR.spir_funcs_release(u)
        LibSparseIR.spir_funcs_release(v)
        LibSparseIR.spir_funcs_release(uhat)
        LibSparseIR.spir_basis_release(basis)
    end

    @testset "Basis Functions Fermionic" begin
        test_basis_functions(LibSparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "Basis Functions Bosonic" begin
        test_basis_functions(LibSparseIR.SPIR_STATISTICS_BOSONIC)
    end
end
