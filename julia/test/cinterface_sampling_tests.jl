# Tests corresponding to test/cpp/cinterface_sampling.cxx
# Comprehensive sampling functionality tests including TauSampling and MatsubaraSampling

@testitem "TauSampling" begin
    using LibSparseIR

    # Helper function to create tau sampling (corresponds to C++ create_tau_sampling)
    function create_tau_sampling(basis::Ptr{LibSparseIR.spir_basis})
        status = Ref{Int32}(0)

        n_tau_points_ref = Ref{Int32}(0)
        points_status = LibSparseIR.spir_basis_get_n_default_taus(basis, n_tau_points_ref)
        @test points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        n_tau_points = n_tau_points_ref[]

        tau_points = Vector{Float64}(undef, n_tau_points)
        get_points_status = LibSparseIR.spir_basis_get_default_taus(basis, tau_points)
        @test get_points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        sampling = LibSparseIR.spir_tau_sampling_new(basis, n_tau_points, tau_points, status)
        @test status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling != C_NULL

        return sampling
    end

    # Test tau sampling constructor (corresponds to C++ test_tau_sampling template)
    function test_tau_sampling(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-15

        # Create basis using kernel and SVE (equivalent to _spir_basis_new)
        kernel_status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = LibSparseIR.spir_sve_result_new(kernel, epsilon, sve_status)
        @test sve_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = LibSparseIR.spir_basis_new(statistics, beta, wmax, kernel, sve, basis_status)
        @test basis_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test basis != C_NULL

        # Get tau points
        n_tau_points_ref = Ref{Int32}(0)
        status = LibSparseIR.spir_basis_get_n_default_taus(basis, n_tau_points_ref)
        @test status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        n_tau_points = n_tau_points_ref[]
        @test n_tau_points > 0

        tau_points_org = Vector{Float64}(undef, n_tau_points)
        tau_status = LibSparseIR.spir_basis_get_default_taus(basis, tau_points_org)
        @test tau_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Create sampling
        sampling_status = Ref{Int32}(0)
        sampling = LibSparseIR.spir_tau_sampling_new(basis, n_tau_points, tau_points_org, sampling_status)
        @test sampling_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling != C_NULL

        # Test getting number of sampling points
        n_points_ref = Ref{Int32}(0)
        points_status = LibSparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]
        @test n_points > 0

        # Test getting sampling points
        tau_points = Vector{Float64}(undef, n_points)
        get_tau_status = LibSparseIR.spir_sampling_get_taus(sampling, tau_points)
        @test get_tau_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Compare tau_points and tau_points_org (corresponds to C++ comparison)
        for i in 1:n_points
            @test tau_points[i] ≈ tau_points_org[i] atol=1e-14
        end

        # Test that matsubara points are not supported for tau sampling
        matsubara_points = Vector{Int64}(undef, n_points)
        matsubara_status = LibSparseIR.spir_sampling_get_matsus(sampling, matsubara_points)
        @test matsubara_status == LibSparseIR.SPIR_NOT_SUPPORTED

        # Clean up
        LibSparseIR.spir_sampling_release(sampling)
        LibSparseIR.spir_basis_release(basis)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(kernel)
    end

    # Test 1D evaluation (corresponds to C++ test_tau_sampling_evaluation_1d_column_major)
    function test_tau_sampling_evaluation_1d(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-10

        # Create basis
        kernel_status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = LibSparseIR.spir_sve_result_new(kernel, epsilon, sve_status)
        @test sve_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = LibSparseIR.spir_basis_new(statistics, beta, wmax, kernel, sve, basis_status)
        @test basis_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Create sampling
        sampling = create_tau_sampling(basis)

        # Get basis and sampling sizes
        basis_size_ref = Ref{Int32}(0)
        size_status = LibSparseIR.spir_basis_get_size(basis, basis_size_ref)
        @test size_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        basis_size = basis_size_ref[]

        n_points_ref = Ref{Int32}(0)
        points_status = LibSparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]

        # Set up parameters for evaluation
        ndim = 1
        dims = Int32[basis_size]
        target_dim = 0

        # Create test coefficients
        coeffs = rand(Float64, basis_size) .- 0.5

        # Test evaluation
        evaluate_output = Vector{Float64}(undef, n_points)
        evaluate_status = LibSparseIR.spir_sampling_eval_dd(
            sampling, LibSparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim, coeffs, evaluate_output)
        @test evaluate_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Test fitting
        fit_output = Vector{Float64}(undef, basis_size)
        fit_status = LibSparseIR.spir_sampling_fit_dd(
            sampling, LibSparseIR.SPIR_ORDER_COLUMN_MAJOR, ndim, dims, target_dim, evaluate_output, fit_output)
        @test fit_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Check round-trip accuracy
        for i in 1:basis_size
            @test fit_output[i] ≈ coeffs[i] atol=1e-10
        end

        # Clean up
        LibSparseIR.spir_sampling_release(sampling)
        LibSparseIR.spir_basis_release(basis)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(kernel)
    end

    @testset "TauSampling Constructor (fermionic)" begin
        test_tau_sampling(LibSparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "TauSampling Constructor (bosonic)" begin
        test_tau_sampling(LibSparseIR.SPIR_STATISTICS_BOSONIC)
    end

    @testset "TauSampling Evaluation 1-dimensional input COLUMN-MAJOR" begin
        test_tau_sampling_evaluation_1d(LibSparseIR.SPIR_STATISTICS_FERMIONIC)
    end
end

@testitem "MatsubaraSampling" begin
    using LibSparseIR

    # Helper function to create matsubara sampling (corresponds to C++ create_matsubara_sampling)
    function create_matsubara_sampling(basis::Ptr{LibSparseIR.spir_basis}, positive_only::Bool)
        status = Ref{Int32}(0)

        n_matsubara_points_ref = Ref{Int32}(0)
        points_status = LibSparseIR.spir_basis_get_nmatuss(basis, positive_only, n_matsubara_points_ref)
        @test points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        n_matsubara_points = n_matsubara_points_ref[]

        smpl_points = Vector{Int64}(undef, n_matsubara_points)
        get_points_status = LibSparseIR.spir_basis_get_matsus(basis, positive_only, smpl_points)
        @test get_points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        sampling = LibSparseIR.spir_matsu_sampling_new(basis, positive_only, n_matsubara_points, smpl_points, status)
        @test status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling != C_NULL

        return sampling
    end

    # Test matsubara sampling constructor (corresponds to C++ test_matsubara_sampling_constructor)
    function test_matsubara_sampling_constructor(statistics::Integer)
        beta = 1.0
        wmax = 10.0
        epsilon = 1e-10

        # Create basis
        kernel_status = Ref{Int32}(0)
        kernel = LibSparseIR.spir_logistic_kernel_new(beta * wmax, kernel_status)
        @test kernel_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        sve_status = Ref{Int32}(0)
        sve = LibSparseIR.spir_sve_result_new(kernel, epsilon, sve_status)
        @test sve_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        basis_status = Ref{Int32}(0)
        basis = LibSparseIR.spir_basis_new(statistics, beta, wmax, kernel, sve, basis_status)
        @test basis_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        # Test with positive_only = false
        n_points_org_ref = Ref{Int32}(0)
        status = LibSparseIR.spir_basis_get_nmatuss(basis, false, n_points_org_ref)
        @test status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        n_points_org = n_points_org_ref[]
        @test n_points_org > 0

        smpl_points_org = Vector{Int64}(undef, n_points_org)
        points_status = LibSparseIR.spir_basis_get_matsus(basis, false, smpl_points_org)
        @test points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        sampling_status = Ref{Int32}(0)
        sampling = LibSparseIR.spir_matsu_sampling_new(basis, false, n_points_org, smpl_points_org, sampling_status)
        @test sampling_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling != C_NULL

        # Test with positive_only = true
        n_points_positive_only_org_ref = Ref{Int32}(0)
        positive_status = LibSparseIR.spir_basis_get_nmatuss(basis, true, n_points_positive_only_org_ref)
        @test positive_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        n_points_positive_only_org = n_points_positive_only_org_ref[]
        @test n_points_positive_only_org > 0

        smpl_points_positive_only_org = Vector{Int64}(undef, n_points_positive_only_org)
        positive_points_status = LibSparseIR.spir_basis_get_matsus(basis, true, smpl_points_positive_only_org)
        @test positive_points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS

        sampling_positive_status = Ref{Int32}(0)
        sampling_positive_only = LibSparseIR.spir_matsu_sampling_new(basis, true, n_points_positive_only_org, smpl_points_positive_only_org, sampling_positive_status)
        @test sampling_positive_status[] == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        @test sampling_positive_only != C_NULL

        # Test getting number of points
        n_points_ref = Ref{Int32}(0)
        get_points_status = LibSparseIR.spir_sampling_get_npoints(sampling, n_points_ref)
        @test get_points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        n_points = n_points_ref[]
        @test n_points > 0

        n_points_positive_only_ref = Ref{Int32}(0)
        get_positive_points_status = LibSparseIR.spir_sampling_get_npoints(sampling_positive_only, n_points_positive_only_ref)
        @test get_positive_points_status == LibSparseIR.SPIR_COMPUTATION_SUCCESS
        n_points_positive_only = n_points_positive_only_ref[]
        @test n_points_positive_only > 0

        # Clean up
        LibSparseIR.spir_sampling_release(sampling_positive_only)
        LibSparseIR.spir_sampling_release(sampling)
        LibSparseIR.spir_basis_release(basis)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(kernel)
    end

    @testset "MatsubaraSampling Constructor (fermionic)" begin
        test_matsubara_sampling_constructor(LibSparseIR.SPIR_STATISTICS_FERMIONIC)
    end

    @testset "MatsubaraSampling Constructor (bosonic)" begin
        test_matsubara_sampling_constructor(LibSparseIR.SPIR_STATISTICS_BOSONIC)
    end
end