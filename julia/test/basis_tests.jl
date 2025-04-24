@testitem "basis" begin
	using LibSparseIR

	β = 0.1
	ωmax = 0.3
	ε = 0.0001

	@testset "Bosonic basis" begin
		bosonic_basis = LibSparseIR.spir_bosonic_finite_temp_basis_new(β, ωmax, ε)
		LibSparseIR.spir_destroy_bosonic_finite_temp_basis(bosonic_basis)
		@test true
	end

	@testset "Fermionic basis" begin
		fermionic_basis = LibSparseIR.spir_fermionic_finite_temp_basis_new(β, ωmax, ε)
		LibSparseIR.spir_destroy_fermionic_finite_temp_basis(fermionic_basis)
		@test true
	end
end
