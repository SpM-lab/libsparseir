! Implementation of the SparseIR Fortran bindings
submodule (sparseir) sparseir_impl
  implicit none

contains

  include '_kernel_impl.f90'

end submodule sparseir_impl 