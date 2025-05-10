! Implementation of the SparseIR Fortran bindings
submodule (sparseir) sparseir_impl
  implicit none

contains

  include '_impl.inc'
  include '_impl_kernel.inc'
  include '_impl_sve_result.inc'
  include '_impl_finite_temp_basis.inc'

end submodule sparseir_impl 