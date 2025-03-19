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
    procedure :: clone => kernel_clone
    procedure :: assign => kernel_assign
    generic :: assignment(=) => assign  ! 代入演算子をオーバーロード
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

contains
  ! Check if kernel is initialized
  function kernel_is_initialized(this) result(initialized)
    class(spir_kernel), intent(in) :: this
    logical :: initialized
    
    initialized = c_associated(this%ptr)
  end function

  ! Clone the kernel object (create a copy)
  function kernel_clone(this) result(copy)
    class(spir_kernel), intent(in) :: this
    type(spir_kernel) :: copy
    
    print *, "Fortran: Cloning kernel with ptr =", this%ptr
    
    if (.not. c_associated(this%ptr)) then
      print *, "Fortran: Source ptr is null, returning uninitialized copy"
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the kernel
    copy%ptr = c_spir_clone_kernel(this%ptr)
    print *, "Fortran: Cloned kernel, new ptr =", copy%ptr
    
    if (.not. c_associated(copy%ptr)) then
      print *, "Fortran: Warning - Clone operation returned NULL pointer"
    end if
  end function

  ! Assignment operator implementation
  subroutine kernel_assign(lhs, rhs)
    class(spir_kernel), intent(inout) :: lhs
    class(spir_kernel), intent(in) :: rhs
    
    print *, "Fortran: Assignment operator called"
    print *, "  LHS ptr =", lhs%ptr
    print *, "  RHS ptr =", rhs%ptr
    
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      print *, "Fortran: Self-assignment detected, nothing to do"
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      print *, "Fortran: Cleaning up existing LHS resource"
      call c_spir_destroy_kernel(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      print *, "Fortran: Cloning RHS to LHS"
      lhs%ptr = c_spir_clone_kernel(rhs%ptr)
    end if
  end subroutine

  ! Finalizer for kernel
  subroutine kernel_finalize(this)
    type(spir_kernel), intent(inout) :: this
    
    if (c_associated(this%ptr)) then
      call c_spir_destroy_kernel(this%ptr)
      this%ptr = c_null_ptr
    end if
  end subroutine

  ! Check if kernel's shared_ptr is assigned (contains a valid C++ object)
  function is_assigned(kernel) result(assigned)
    type(spir_kernel), intent(in) :: kernel
    logical :: assigned
    
    !print *, "Fortran: Checking if kernel's shared_ptr is assigned"
    
    if (.not. c_associated(kernel%ptr)) then
      !print *, "Fortran: Pointer is not associated at C level"
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_kernel(kernel%ptr) /= 0)
    !print *, "Fortran: C++ shared_ptr is_assigned =", assigned
  end function

  ! Wrapper for kernel creation
  function spir_logistic_kernel_new(lambda) result(kernel)
    real(c_double), intent(in) :: lambda
    type(spir_kernel) :: kernel
    
    print *, "Calling C function to create logistic kernel with lambda =", lambda
    kernel%ptr = c_spir_logistic_kernel_new(lambda)
    print *, "C function returned kernel ptr =", kernel%ptr
    
    if (.not. c_associated(kernel%ptr)) then
      print *, "Warning: C function returned NULL pointer"
    end if
  end function

  ! Wrapper for kernel domain
  function spir_kernel_domain(kernel, xmin, xmax, ymin, ymax) result(stat)
    type(spir_kernel), intent(in) :: kernel
    real(c_double), intent(out) :: xmin, xmax, ymin, ymax
    integer(c_int) :: stat
    
    if (.not. kernel%is_initialized()) then
      print *, "Error: Kernel not initialized in spir_kernel_domain"
      stat = -1
      return
    end if
    
    print *, "Calling C function c_spir_kernel_domain with ptr =", kernel%ptr
    print *, "Is pointer associated? ", c_associated(kernel%ptr)
    
    ! Initialize output variables
    xmin = 0.0_c_double
    xmax = 0.0_c_double
    ymin = 0.0_c_double
    ymax = 0.0_c_double
    
    stat = c_spir_kernel_domain(kernel%ptr, xmin, xmax, ymin, ymax)
    print *, "C function c_spir_kernel_domain returned status =", stat
  end function

end module sparseir 