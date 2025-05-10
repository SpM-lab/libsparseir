import clang.cindex
from clang.cindex import Index, TypeKind, CursorKind
import re


def map_c_type_to_fortran(ctype):
    """Map a C type to a Fortran iso_c_binding type."""
    kind = ctype.kind
    if kind == TypeKind.VOID:
        return 'subroutine'
    elif kind == TypeKind.DOUBLE:
        return 'real(c_double), value'
    elif kind == TypeKind.INT or kind == TypeKind.INT:
        return 'integer(c_int), value'
    elif kind == TypeKind.UINT or kind == TypeKind.UINT:
        return 'integer(c_int), value'
    elif kind == TypeKind.POINTER:
        pointee = ctype.get_pointee()
        type_name = pointee.get_canonical().spelling
        is_const = 'const' in type_name
        if pointee.kind == TypeKind.RECORD:
            return 'type(c_ptr), intent(in)' if is_const else 'type(c_ptr), intent(out)'
        elif pointee.kind == TypeKind.DOUBLE:
            return 'real(c_double), intent(out)'
        elif pointee.kind == TypeKind.COMPLEX:
            return 'complex(c_double_complex), intent(out)'
        elif pointee.kind == TypeKind.INT:
            return 'integer(c_int), intent(out)'
        elif pointee.kind == TypeKind.ENUM:
            return 'integer(c_int), intent(out)'
        elif pointee.kind == TypeKind.ELABORATED:
            if type_name in ["spir_statistics_type", "spir_order_type"]:
                return 'integer(c_int), intent(out)'
            return 'type(c_ptr), intent(in)' if is_const else 'type(c_ptr), intent(out)'
        else:
            return 'type(c_ptr)'
    elif kind == TypeKind.COMPLEX:
        return 'complex(c_double_complex), value'
    elif kind == TypeKind.ENUM:
        return 'integer(c_int), value'
    elif kind == TypeKind.ELABORATED:
        type_name = ctype.get_canonical().spelling
        if type_name in ["spir_statistics_type", "spir_order_type"]:
            return 'integer(c_int), value'
        return 'type(c_ptr)'
    else:
        return 'type(c_ptr)'  # default fallback


def generate_fortran_interface(cursor, types):
    """Generate Fortran interface code from a C function declaration."""
    if cursor.kind != CursorKind.FUNCTION_DECL:
        return "", ""

    func_name = cursor.spelling
    fortran_func_name = f"c_{func_name}"
    result_type = map_c_type_to_fortran(cursor.result_type)

    args = []
    decls = []
    fortran_args = []
    fortran_body = []
    for arg in cursor.get_arguments():
        name = arg.spelling or 'arg'
        ftype = map_c_type_to_fortran(arg.type)
        args.append(name)
        decls.append(f"  {ftype} :: {name}")
        
        # Convert C types to Fortran types for friendly interface
        if 'type(c_ptr)' in ftype:
            # Get the actual type from the C type
            pointee = arg.type.get_pointee()
            type_name = pointee.get_canonical().spelling
            #print(f"Pointee type: {type_name}")
            
            # Clean up type name: remove 'const' and 'struct _'
            type_name = type_name.replace('const ', '')
            type_name = type_name.replace('struct _', '')
            
            # Convert C type name to Fortran type name
            if type_name.startswith('spir_'):
                type_name = type_name  # Already has spir_ prefix
            else:
                type_name = f"spir_{type_name}"
            
            if type_name in types:
                # For output parameters, use intent(inout)
                if 'intent(out)' in ftype:
                    fortran_args.append(f"  type({type_name}), intent(inout) :: {name}")
                else:
                    fortran_args.append(f"  type({type_name}), intent(in) :: {name}")
                fortran_body.append(f"  {name}%%ptr")  # Use %% to escape %
            else:
                fortran_args.append(f"  {ftype} :: {name}")
                fortran_body.append(f"  {name}")
        else:
            fortran_args.append(f"  {ftype} :: {name}")
            fortran_body.append(f"  {name}")

    arglist = ', '.join(args)
    decl_lines = '\n'.join(decls)
    fortran_decl_lines = '\n'.join(fortran_args)
    fortran_body_lines = ',\n'.join(fortran_body)

    # Generate C binding interface
    if result_type == 'subroutine':
        c_binding = f"""
subroutine {fortran_func_name}({arglist}) bind(c, name="{func_name}")
  use iso_c_binding
{decl_lines}
end subroutine
""".strip()
    else:
        result_decl = f"  {result_type.split(',')[0]} :: {fortran_func_name}"
        c_binding = f"""
function {fortran_func_name}({arglist}) bind(c, name="{func_name}") result({fortran_func_name})
  use iso_c_binding
{decl_lines}
{result_decl}
end function
""".strip()

    # Generate Fortran-friendly interface
    # Remove spir_ prefix for Fortran interface
    fortran_name = func_name[5:] if func_name.startswith('spir_') else func_name
    if result_type == 'subroutine':
        fortran_interface = f"""
subroutine {fortran_name}({arglist})
  use iso_c_binding
{fortran_decl_lines}
  call {fortran_func_name}({fortran_body_lines})
end subroutine
""".strip()
    else:
        fortran_interface = f"""
function {fortran_name}({arglist}) result({fortran_name})
  use iso_c_binding
{fortran_decl_lines}
  {fortran_name} = {fortran_func_name}({fortran_body_lines})
end function
""".strip()

    #print("--------------------------------")
    #print("C binding:")
    #print(c_binding)
    #print("--------------------------------")
    #print("Fortran interface:")
    #print(fortran_interface)
    return c_binding, fortran_interface


def generate_fortran_type_definition(type_name):
    """Generate Fortran type definition for an C type."""
    return f"""  type :: {type_name}
    type(c_ptr) :: ptr = c_null_ptr
  contains
    generic :: assignment(=) => assign_{type_name}  ! Overload assignment operator
    final :: spir_destroy_{type_name}
  end type

  ! Assignment operator implementation
  module procedure assign_{type_name}
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

"""


def generate_proc_interfaces(type_name):
    """Generate Fortran procedure interfaces for a type."""
    fortran_type_name = f"{type_name}"
    return f"""  ! Check if {type_name} is initialized
  module function {type_name}_is_initialized(this) result(initialized)
    class({fortran_type_name}), intent(in) :: this
    logical :: initialized
  end function

  ! Clone the {type_name} object (create a copy)
  module function {type_name}_clone(this) result(copy)
    class({fortran_type_name}), intent(in) :: this
    type({fortran_type_name}) :: copy
  end function

  ! Assignment operator implementation
  module subroutine {type_name}_assign(lhs, rhs)
    class({fortran_type_name}), intent(inout) :: lhs
    class({fortran_type_name}), intent(in) :: rhs
  end subroutine

  ! Finalizer for {type_name}
  module subroutine {type_name}_finalize(this)
    type({fortran_type_name}), intent(inout) :: this
  end subroutine

  ! Check if {type_name}'s shared_ptr is assigned
  ! (contains a valid C++ object)
  module function {type_name}_is_assigned({type_name}) result(assigned)
    type({fortran_type_name}), intent(in) :: {type_name}
    logical :: assigned
  end function

"""


def find_c_types(header_path):
    """Find all types declared with DECLARE_OPAQUE_TYPE macro."""
    types = set()
    pattern = re.compile(r'DECLARE_OPAQUE_TYPE\(([a-zA-Z0-9_]+)\)')
    
    with open(header_path, 'r') as f:
        content = f.read()
    
    # Remove #define section to avoid matching the macro definition itself
    content = re.sub(r'#define.*?\\\n.*?\\\n.*?\\\n.*?\\\n', '', content, flags=re.DOTALL)
    
    # Find all matches
    for match in pattern.finditer(content):
        type_name = match.group(1)
        # Skip if the type is not declared with DECLARE_OPAQUE_TYPE
        if not f"DECLARE_OPAQUE_TYPE({type_name})" in content:
            continue
        # Add spir_ prefix to type name
        types.add(f"spir_{type_name}")
    
    # Print found types
    print("Found types declared with DECLARE_OPAQUE_TYPE:")
    for type_name in sorted(types):
        print(f"  - {type_name}")
    
    return sorted(list(types))


def generate_proc_inc(types):
    """Generate _proc.inc file content."""
    content = []
    
    # Generate is_initialized interface
    content.append("interface is_initialized")
    for type_name in types:
        content.append(f"""  ! Check if {type_name} is initialized
  module function {type_name}_is_initialized(this) result(initialized)
    class({type_name}), intent(in) :: this
    logical :: initialized
  end function""")
    content.append("end interface\n")

    # Generate clone interface
    content.append("interface clone")
    for type_name in types:
        content.append(f"""  ! Clone the {type_name} object (create a copy)
  module function {type_name}_clone(this) result(copy)
    class({type_name}), intent(in) :: this
    type({type_name}) :: copy
  end function""")
    content.append("end interface\n")

    # Generate assign interface
    content.append("interface assign")
    for type_name in types:
        content.append(f"""  ! Assignment operator implementation
  module subroutine {type_name}_assign(lhs, rhs)
    class({type_name}), intent(inout) :: lhs
    class({type_name}), intent(in) :: rhs
  end subroutine""")
    content.append("end interface\n")

    # Generate finalize interface
    content.append("interface finalize")
    for type_name in types:
        content.append(f"""  ! Finalizer for {type_name}
  module subroutine {type_name}_finalize(this)
    type({type_name}), intent(inout) :: this
  end subroutine""")
    content.append("end interface\n")

    # Generate is_assigned interface
    content.append("interface is_assigned")
    for type_name in types:
        content.append(f"""  ! Check if {type_name}'s shared_ptr is assigned
  ! (contains a valid C++ object)
  module function {type_name}_is_assigned(this) result(assigned)
    type({type_name}), intent(in) :: this
    logical :: assigned
  end function""")
    content.append("end interface\n")

    # Join with double newlines between interfaces
    return "\n\n".join(content)


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_fortran_interfaces.py spir_sample.h")
        return

    header_path = sys.argv[1]
    index = Index.create()
    tu = index.parse(header_path, args=['-x', 'c', '-std=c99'])

    # Find C types first
    types = find_c_types(header_path)

    # Generate interface bindings
    c_interfaces = []
    fortran_interfaces = []
    print("\nFound C functions:")
    for cursor in tu.cursor.get_children():
        if cursor.location.file and cursor.location.file.name == header_path:
            if cursor.kind == CursorKind.FUNCTION_DECL:
                print(f"  - {cursor.spelling}")
            c_binding, fortran_interface = generate_fortran_interface(cursor, types)
            if c_binding:
                c_interfaces.append(c_binding)
            if fortran_interface:
                fortran_interfaces.append(fortran_interface)

    # Write C binding interfaces
    with open("_cbinding.inc", 'w') as f:
        f.write("! Autogenerated Fortran interfaces for " + header_path + "\n")
        for code in c_interfaces:
            f.write(code + "\n\n")

    # Write Fortran-friendly interfaces
    with open("_fortran_funcs.inc", 'w') as f:
        f.write("! Autogenerated Fortran-friendly interfaces\n")
        for code in fortran_interfaces:
            f.write(code + "\n\n")

    # Write type definitions
    with open("_fortran_types.inc", 'w') as f:
        f.write("! Autogenerated Fortran type definitions\n")
        for type_name in types:
            f.write(generate_fortran_type_definition(type_name))

    # Generate and write _proc.inc
    #with open("_proc.inc", 'w') as f:
        #f.write("! Autogenerated Fortran procedure interfaces\n")
        #f.write(generate_proc_inc(types))
    
    print("\nGenerated Fortran interfaces written to _cbinding.inc")
    print("Generated Fortran-friendly interfaces written to _fortran_funcs.inc")
    print("Generated Fortran type definitions written to _fortran_types.inc")
    #print("Generated Fortran procedure interfaces written to _proc.inc")


if __name__ == "__main__":
    main()
