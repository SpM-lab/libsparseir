  type :: spir_sve_result
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => sve_result_is_initialized
    procedure :: clone => sve_result_clone
    procedure :: assign => sve_result_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: sve_result_finalize
  end type
