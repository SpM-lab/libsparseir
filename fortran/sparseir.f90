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
  public :: is_assigned

  ! Enumeration types
  enum, bind(c)
    enumerator :: SPIR_STATISTICS_FERMIONIC = 1
    enumerator :: SPIR_STATISTICS_BOSONIC = 0
  end enum

  ! Opaque handle for kernel object
  type :: spir_kernel
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => kernel_is_initialized
    procedure :: clone => kernel_clone
    procedure :: assign => kernel_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
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

    ! Clone an existing kernel
    function c_spir_clone_kernel(src) bind(c, name='spir_clone_kernel')
      import c_ptr
      type(c_ptr), value :: src
      type(c_ptr) :: c_spir_clone_kernel
    end function

    ! Get the domain of a kernel
    function c_spir_kernel_domain(k, xmin, xmax, ymin, ymax) bind(c, name='spir_kernel_domain')
      import c_ptr, c_double, c_int
      type(c_ptr), value :: k
      real(c_double) :: xmin, xmax, ymin, ymax
      integer(c_int) :: c_spir_kernel_domain
    end function

    ! Check if kernel's shared_ptr is assigned
    function c_spir_is_assigned_kernel(k) bind(c, name='spir_is_assigned_kernel')
      import c_ptr, c_int
      type(c_ptr), value :: k
      integer(c_int) :: c_spir_is_assigned_kernel
    end function

    ! Destroy a kernel
    subroutine c_spir_destroy_kernel(k) bind(c, name='spir_destroy_kernel')
      import c_ptr
      type(c_ptr), value :: k
    end subroutine
  end interface

  ! Interface declarations for module procedures
  interface
    ! Check if kernel is initialized
    module function kernel_is_initialized(this) result(initialized)
      class(spir_kernel), intent(in) :: this
      logical :: initialized
    end function

    ! Clone the kernel object (create a copy)
    module function kernel_clone(this) result(copy)
      class(spir_kernel), intent(in) :: this
      type(spir_kernel) :: copy
    end function

    ! Assignment operator implementation
    module subroutine kernel_assign(lhs, rhs)
      class(spir_kernel), intent(inout) :: lhs
      class(spir_kernel), intent(in) :: rhs
    end subroutine

    ! Finalizer for kernel
    module subroutine kernel_finalize(this)
      type(spir_kernel), intent(inout) :: this
    end subroutine

    ! Check if kernel's shared_ptr is assigned (contains a valid C++ object)
    module function is_assigned(kernel) result(assigned)
      type(spir_kernel), intent(in) :: kernel
      logical :: assigned
    end function

    ! Wrapper for kernel creation
    module function spir_logistic_kernel_new(lambda) result(kernel)
      real(c_double), intent(in) :: lambda
      type(spir_kernel) :: kernel
    end function

    ! Wrapper for kernel domain
    module function spir_kernel_domain(kernel, xmin, xmax, ymin, ymax) result(stat)
      type(spir_kernel), intent(in) :: kernel
      real(c_double), intent(out) :: xmin, xmax, ymin, ymax
      integer(c_int) :: stat
    end function
  end interface

end module sparseir