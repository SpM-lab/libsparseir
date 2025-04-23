@testitem "basis" begin
	using LibSparseIR_C_API
	b = spir_bosonic_finite_temp_basis_new(0.1, 0.3, 0.0001)
	@test true
end
