@testitem "kernel" begin
	using LibSparseIR

	@testset "Logistic kernel" begin
		k = LibSparseIR.spir_logistic_kernel_new(9.0)
		LibSparseIR.spir_release_logistic_kernel(k)
		@test true
	end

	@testset "Regularized Bose kernel" begin
		k = LibSparseIR.spir_reg_bose_kernel_new(10.0)
		LibSparseIR.spir_release_reg_bose_kernel(k)
		@test true
	end

	@testset "Kernel domain" begin
		k = LibSparseIR.spir_logistic_kernel_new(9.0)

		xmin = Ref(0.0)
		xmax = Ref(0.0)
		ymin = Ref(0.0)
		ymax = Ref(0.0)
		status = LibSparseIR.spir_kernel_domain(k, xmin, xmax, ymin, ymax)
		@test status == 0
		@test xmin[] == -1.0
		@test xmax[] == 1.0
		@test ymin[] == -1.0
		@test ymax[] == 1.0
	end
end
