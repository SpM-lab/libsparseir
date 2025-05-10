! Assignment operator interfaces
interface assignment(=)
  module procedure assign_spir_dlr
  module procedure assign_spir_finite_temp_basis
  module procedure assign_spir_funcs
  module procedure assign_spir_kernel
  module procedure assign_spir_matsubara_funcs
  module procedure assign_spir_sampling
  module procedure assign_spir_sve_result
end interface