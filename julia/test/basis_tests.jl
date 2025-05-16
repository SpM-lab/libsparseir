@testitem "basis" begin
	using LibSparseIR

	β = 2.0
	ωmax = 5.0
	ε = 1e-6

	@testset "Bosonic basis" begin
		basis = LibSparseIR.spir_bosonic_finite_temp_basis_new(β, ωmax, ε)
		@test basis != C_NULL
		basis_size_ref = Ref{Cint}(0)
		status = LibSparseIR.spir_bosonic_finite_temp_basis_get_size(basis, basis_size_ref)
		@test status == 0
		basis_size = basis_size_ref[]
		@test basis_size > 0
		LibSparseIR.spir_release_bosonic_finite_temp_basis(basis)
		@test true
	end

	@testset "Fermionic basis" begin
		basis = LibSparseIR.spir_fermionic_finite_temp_basis_new(β, ωmax, ε)
		@test basis != C_NULL
		basis_size_ref = Ref{Cint}(0)
		status = LibSparseIR.spir_fermionic_finite_temp_basis_get_size(basis, basis_size_ref)
		@test status == 0
		basis_size = basis_size_ref[]
		@test basis_size > 0
		LibSparseIR.spir_release_fermionic_finite_temp_basis(basis)
		@test true
	end
end
