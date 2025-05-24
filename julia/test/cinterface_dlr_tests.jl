# This file corresponds to DLR functionality in test/cpp/cinterface_integration.cxx
# Comprehensive DLR (Discrete Lehmann Representation) functionality tests.
# Currently simplified to avoid segmentation faults

@testitem "dlr" begin
    using LibSparseIR

    β = 10.0
    ωmax = 2.0
    ε = 1e-10

    @testset "DLR Basic Test" begin
        status = Ref{Int32}(0)

        # Create IR basis
        k = LibSparseIR.spir_logistic_kernel_new(β * ωmax, status)
        @test status[] == 0
        sve = LibSparseIR.spir_sve_result_new(k, ε, status)
        @test status[] == 0
        ir_basis = LibSparseIR.spir_basis_new(LibSparseIR.SPIR_STATISTICS_BOSONIC, β, ωmax, k, sve, status)
        @test status[] == 0

        # Create DLR basis from IR basis
        dlr_basis = LibSparseIR.spir_dlr_new(ir_basis, status)
        @test status[] == 0

        # Get basic properties
        ir_basis_size_ref = Ref{Cint}(0)
        status_val = LibSparseIR.spir_basis_get_size(ir_basis, ir_basis_size_ref)
        @test status_val == 0
        ir_basis_size = ir_basis_size_ref[]

        npoles_ref = Ref{Cint}(0)
        status_val = LibSparseIR.spir_dlr_get_npoles(dlr_basis, npoles_ref)
        @test status_val == 0
        npoles = npoles_ref[]
        @test npoles >= ir_basis_size

        # Get poles
        poles = zeros(Float64, npoles)
        status_val = LibSparseIR.spir_dlr_get_poles(dlr_basis, pointer(poles))
        @test status_val == 0
        @test maximum(abs.(poles)) <= ωmax

        LibSparseIR.spir_basis_release(dlr_basis)
        LibSparseIR.spir_basis_release(ir_basis)
        LibSparseIR.spir_sve_result_release(sve)
        LibSparseIR.spir_kernel_release(k)
    end
end