! Assignment operator interfaces
interface assignment(=)
  module procedure assign_spir_basis
  module procedure assign_spir_funcs
  module procedure assign_spir_kernel
  module procedure assign_spir_sampling
  module procedure assign_spir_sve_result
end interface