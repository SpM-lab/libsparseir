! Fortran interface for the SparseIR library
! This module provides Fortran bindings to the SparseIR C API using iso_c_binding
module sparseir
  use, intrinsic :: iso_c_binding
  implicit none
  private

  ! Export public interfaces
  public :: spir_kernel
  public :: spir_logistic_kernel_new
  public :: spir_kernel_domain
  public :: spir_destroy_kernel

  ! Version constants
  integer, parameter, public :: SPIR_VERSION_MAJOR = 0
  integer, parameter, public :: SPIR_VERSION_MINOR = 1
  integer, parameter, public :: SPIR_VERSION_PATCH = 0

  ! Enumeration types
  enum, bind(c)
    enumerator :: SPIR_STATISTICS_FERMIONIC = 1
    enumerator :: SPIR_STATISTICS_BOSONIC = 0
  end enum

  enum, bind(c)
    enumerator :: SPIR_ORDER_COLUMN_MAJOR = 1
    enumerator :: SPIR_ORDER_ROW_MAJOR = 0
  end enum

  ! Opaque handle for kernel object
  type :: spir_kernel
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => kernel_is_initialized
    final :: kernel_finalize
  end type

  ! Interface declarations for C API functions
  interface
    ! Create a new logistic kernel
    function c_spir_logistic_kernel_new(lambda) bind(c, name='spir_logistic_kernel_new')
      import c_ptr, c_double
      real(c_double), value :: lambda
      type(c_ptr) :: c_spir_logistic_kernel_new
    end function

    ! Get the domain of a kernel
    function c_spir_kernel_domain(k, xmin, xmax, ymin, ymax) bind(c, name='spir_kernel_domain')
      import c_ptr, c_double, c_int
      type(c_ptr), value :: k
      real(c_double) :: xmin, xmax, ymin, ymax
      integer(c_int) :: c_spir_kernel_domain
    end function

    ! Destroy a kernel
    subroutine c_spir_destroy_kernel(k) bind(c, name='spir_destroy_kernel')
      import c_ptr
      type(c_ptr), value :: k
    end subroutine
  end interface

contains
  ! Check if kernel is initialized
  function kernel_is_initialized(this) result(initialized)
    class(spir_kernel), intent(in) :: this
    logical :: initialized
    
    initialized = c_associated(this%ptr)
  end function

  ! Finalizer for kernel
  subroutine kernel_finalize(this)
    type(spir_kernel), intent(inout) :: this
    
    if (this%is_initialized()) then
      call spir_destroy_kernel(this)
    end if
  end subroutine

  ! Wrapper for kernel creation
  function spir_logistic_kernel_new(lambda) result(kernel)
    real(c_double), intent(in) :: lambda
    type(spir_kernel) :: kernel
    
    kernel%ptr = c_spir_logistic_kernel_new(lambda)
  end function

  ! Wrapper for kernel domain
  function spir_kernel_domain(kernel, xmin, xmax, ymin, ymax) result(stat)
    type(spir_kernel), intent(in) :: kernel
    real(c_double), intent(out) :: xmin, xmax, ymin, ymax
    integer(c_int) :: stat
    
    if (.not. kernel%is_initialized()) then
      stat = -1
      return
    end if
    
    stat = c_spir_kernel_domain(kernel%ptr, xmin, xmax, ymin, ymax)
  end function

  ! Wrapper for kernel destruction
  subroutine spir_destroy_kernel(kernel)
    type(spir_kernel), intent(inout) :: kernel
    
    if (kernel%is_initialized()) then
      call c_spir_destroy_kernel(kernel%ptr)
      kernel%ptr = c_null_ptr
    end if
  end subroutine

end module sparseir 