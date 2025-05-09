#!/usr/bin/env python3

import sys
import os


def show_help():
    """Show help message."""
    print("""Usage: python generate_binding.py <type_name>

Generate Fortran binding files for the specified C type.

Arguments:
  type_name    Name of the C type (e.g., kernel)

Example:
  python generate_binding.py kernel    # Generate _kernel_*.f90 files

Generated files:
  - _{type_name}_type.f90   : Type definition
  - _{type_name}_capi.f90   : C API interface
  - _{type_name}_proc.f90   : Procedure interfaces
  - _{type_name}_impl.f90   : Procedure implementations
""")


def generate_type_files(type_name):
    """Generate Fortran binding files for the given C type."""
    
    # Generate _type_type.f90
    with open(f'_{type_name}_type.f90', 'w') as f:
        f.write(f"""  type :: spir_{type_name}
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => {type_name}_is_initialized
    procedure :: clone => {type_name}_clone
    procedure :: assign => {type_name}_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: {type_name}_finalize
  end type
""")

    # Generate _type_capi.f90
    with open(f'_{type_name}_capi.f90', 'w') as f:
        f.write(f"""module {type_name}_capi
  use, intrinsic :: iso_c_binding
  implicit none

  interface
    ! Clone an existing {type_name}
    function c_spir_clone_{type_name}(src) &
        bind(c, name='spir_clone_{type_name}')
      import c_ptr
      type(c_ptr), value :: src
      type(c_ptr) :: c_spir_clone_{type_name}
    end function

    ! Check if {type_name}'s shared_ptr is assigned
    function c_spir_is_assigned_{type_name}(k) &
        bind(c, name='spir_is_assigned_{type_name}')
      import c_ptr, c_int
      type(c_ptr), value :: k
      integer(c_int) :: c_spir_is_assigned_{type_name}
    end function

    ! Destroy a {type_name}
    subroutine c_spir_destroy_{type_name}(k) &
        bind(c, name='spir_destroy_{type_name}')
      import c_ptr
      type(c_ptr), value :: k
    end subroutine
  end interface

end module {type_name}_capi
""")

    # Generate _type_proc.f90
    with open(f'_{type_name}_proc.f90', 'w') as f:
        f.write(f"""    ! Check if {type_name} is initialized
    module function {type_name}_is_initialized(this) result(initialized)
      class(spir_{type_name}), intent(in) :: this
      logical :: initialized
    end function

    ! Clone the {type_name} object (create a copy)
    module function {type_name}_clone(this) result(copy)
      class(spir_{type_name}), intent(in) :: this
      type(spir_{type_name}) :: copy
    end function

    ! Assignment operator implementation
    module subroutine {type_name}_assign(lhs, rhs)
      class(spir_{type_name}), intent(inout) :: lhs
      class(spir_{type_name}), intent(in) :: rhs
    end subroutine

    ! Finalizer for {type_name}
    module subroutine {type_name}_finalize(this)
      type(spir_{type_name}), intent(inout) :: this
    end subroutine
""")

    # Generate _type_impl.f90
    with open(f'_{type_name}_impl.f90', 'w') as f:
        f.write(f"""  ! Check if {type_name} is initialized
  module procedure {type_name}_is_initialized
    initialized = c_associated(this%ptr)
  end procedure

  ! Clone the {type_name} object (create a copy)
  module procedure {type_name}_clone
    if (.not. c_associated(this%ptr)) then
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the {type_name}
    copy%ptr = c_spir_clone_{type_name}(this%ptr)
  end procedure

  ! Assignment operator implementation
  module procedure {type_name}_assign
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      call c_spir_destroy_{type_name}(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      lhs%ptr = c_spir_clone_{type_name}(rhs%ptr)
    end if
  end procedure

  ! Finalizer for {type_name}
  module procedure {type_name}_finalize
    if (c_associated(this%ptr)) then
      call c_spir_destroy_{type_name}(this%ptr)
      this%ptr = c_null_ptr
    end if
  end procedure
""")


def main():
    if len(sys.argv) != 2:
        show_help()
        sys.exit(1)

    type_name = sys.argv[1]
    generate_type_files(type_name)


if __name__ == "__main__":
    main() 