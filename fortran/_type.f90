

  type :: spir_kernel
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => kernel_is_initialized
    procedure :: clone => kernel_clone
    procedure :: assign => kernel_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: kernel_finalize
  end type

  type :: spir_sve_result
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => sve_result_is_initialized
    procedure :: clone => sve_result_clone
    procedure :: assign => sve_result_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: sve_result_finalize
  end type

