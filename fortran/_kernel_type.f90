  type :: spir_kernel
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => kernel_is_initialized
    procedure :: clone => kernel_clone
    procedure :: assign => kernel_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: kernel_finalize
  end type