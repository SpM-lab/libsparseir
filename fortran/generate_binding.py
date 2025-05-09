#!/usr/bin/env python3

import sys
import json


def show_help():
    """Show help message."""
    print("""Usage: python generate_binding.py [config_file]

Generate Fortran binding files for all C types.

Arguments:
  config_file    Path to JSON configuration file (default: types.json)

Generated files:
  - _type.f90   : Type definitions
  - _capi.f90   : C API interfaces
  - _proc.f90   : Procedure interfaces
  - _impl.f90   : Procedure implementations
""")


def load_types(config_file):
    """Load types from configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config.get('types', [])
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in configuration file '{config_file}'.")
        sys.exit(1)


def generate_type_files(types):
    """Generate Fortran binding files for all C types."""
    # Generate _type.f90
    with open('_type.f90', 'w') as f:
        f.write("\n\n")
        for type_name in types:
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

    # Generate _public.f90
    with open('_public.f90', 'w') as f:
        f.write("\n\n")
        for type_name in types:
            f.write(f"""  public :: spir_{type_name}
""")

    # Generate _capi.f90
    with open('_capi.f90', 'w') as f:
        for type_name in types:
            f.write(f"""  ! Clone an existing {type_name}
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

""")

    # Generate _proc.f90
    with open('_proc.f90', 'w') as f:
        f.write("interface is_initialized\n")
        for type_name in types:
            f.write(f"""  ! Check if {type_name} is initialized
  module function {type_name}_is_initialized(this) result(initialized)
    class(spir_{type_name}), intent(in) :: this
    logical :: initialized
  end function

""")
        f.write("end interface\n\n")

        f.write("interface clone\n")
        for type_name in types:
            f.write(f"""  ! Clone the {type_name} object (create a copy)
  module function {type_name}_clone(this) result(copy)
    class(spir_{type_name}), intent(in) :: this
    type(spir_{type_name}) :: copy
  end function

  """)
        f.write("end interface\n\n")

        f.write("interface assign\n")
        for type_name in types:
            f.write(f"""  ! Assignment operator implementation
  module subroutine {type_name}_assign(lhs, rhs)
    class(spir_{type_name}), intent(inout) :: lhs
    class(spir_{type_name}), intent(in) :: rhs
  end subroutine

""")
        f.write("end interface\n\n")

        f.write("interface finalize\n")
        for type_name in types:
            f.write(f"""  ! Finalizer for {type_name}
  module subroutine {type_name}_finalize(this)
    type(spir_{type_name}), intent(inout) :: this
  end subroutine

""")
        f.write("end interface\n\n")

        f.write("interface is_assigned\n")
        for type_name in types:
            f.write(
                f"""  ! Check if {type_name}'s shared_ptr is assigned
  ! (contains a valid C++ object)
  module function {type_name}_is_assigned({type_name}) result(assigned)
    type(spir_{type_name}), intent(in) :: {type_name}
    logical :: assigned
  end function

""")
        f.write("end interface\n\n")


    # Generate _impl.f90
    with open('_impl.f90', 'w') as f:
        # Break long line into multiple lines
        for type_name in types:
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

  ! Check if {type_name}'s shared_ptr is assigned (contains a valid C++ object)
  module procedure {type_name}_is_assigned
    if (.not. c_associated({type_name}%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_{type_name}({type_name}%ptr) /= 0)
  end procedure

""")


def main():
    if len(sys.argv) > 2:
        show_help()
        sys.exit(1)

    config_file = sys.argv[1] if len(sys.argv) == 2 else 'types.json'
    types = load_types(config_file)
    
    if not types:
        print("Error: No types found in configuration file.")
        sys.exit(1)

    generate_type_files(types)


if __name__ == "__main__":
    main() 