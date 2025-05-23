@testitem "basis" begin
	using LibSparseIR

	β = 2.0
	ωmax = 5.0
	ε = 1e-6
	
	STATISTICS = [
		LibSparseIR.SPIR_STATISTICS_BOSONIC,
		LibSparseIR.SPIR_STATISTICS_FERMIONIC,
	]

	@testset "Statistics $(stat) " for stat in STATISTICS
		status = Ref{Int32}(0)
		k = LibSparseIR.spir_logistic_kernel_new(β * ωmax, status)
		@test status[] == 0
		sve = LibSparseIR.spir_sve_result_new(k, ε, status)
		@test status[] == 0
		basis = LibSparseIR.spir_basis_new(stat, β, ωmax, k, sve, status)
		@test status[] == 0
		@test basis != C_NULL
		basis_size_ref = Ref{Cint}(0)
		status = LibSparseIR.spir_basis_get_size(basis, basis_size_ref)
		@test status == 0
		basis_size = basis_size_ref[]
		@test basis_size > 0
		LibSparseIR.spir_basis_release(basis)
		@test true
	end
end
