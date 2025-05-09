! Fortran interface for the SparseIR library
! This module provides Fortran bindings to the SparseIR C API using iso_c_binding
module sparseir
  use, intrinsic :: iso_c_binding
  implicit none
  private

  ! Export public interfaces
  public :: is_assigned

  include '_public.f90'
  include '_public_kernel.f90'
  include '_public_sve_result.f90'

  ! Enumeration types
  enum, bind(c)
    enumerator :: SPIR_STATISTICS_FERMIONIC = 1
    enumerator :: SPIR_STATISTICS_BOSONIC = 0
  end enum

  include '_type.f90'

  ! Interface declarations for C API functions
  interface
    include '_capi.f90'

    include '_kernel_capi.f90'
    include '_sve_result_capi.f90'
  end interface

  ! Type generic procedures
  include '_proc.f90'

  ! Type specific procedures
  interface
    include '_kernel_proc.f90'
    include '_sve_result_proc.f90'
  end interface

end module sparseir