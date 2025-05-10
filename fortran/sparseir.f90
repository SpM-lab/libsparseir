! Fortran interface for the SparseIR library
! This module provides Fortran bindings to the SparseIR C API using iso_c_binding
module sparseir
  use, intrinsic :: iso_c_binding
  implicit none
  private

  ! Export public interfaces
  include '_fortran_types_public.inc'
  include '_cbinding_public.inc'
  public :: SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC

  ! Enumeration types
  enum, bind(c)
    enumerator :: SPIR_STATISTICS_FERMIONIC = 1
    enumerator :: SPIR_STATISTICS_BOSONIC = 0
  end enum

  ! Type definitions
  include '_fortran_types.inc'

  ! Assignment operator interfaces
  include '_fortran_assign.inc'

  ! C bindings
  interface
    include '_cbinding.inc'
  end interface

  contains
    ! Type implementations
    include '_impl_types.inc'

end module sparseir