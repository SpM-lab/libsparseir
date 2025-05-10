

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

  type :: spir_funcs
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => funcs_is_initialized
    procedure :: clone => funcs_clone
    procedure :: assign => funcs_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: funcs_finalize
  end type

  type :: spir_matsubara_funcs
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => matsubara_funcs_is_initialized
    procedure :: clone => matsubara_funcs_clone
    procedure :: assign => matsubara_funcs_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: matsubara_funcs_finalize
  end type

  type :: spir_finite_temp_basis
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => finite_temp_basis_is_initialized
    procedure :: clone => finite_temp_basis_clone
    procedure :: assign => finite_temp_basis_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: finite_temp_basis_finalize
  end type

  type :: spir_sampling
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => sampling_is_initialized
    procedure :: clone => sampling_clone
    procedure :: assign => sampling_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: sampling_finalize
  end type

  type :: spir_dlr
    type(c_ptr) :: ptr = c_null_ptr
  contains
    procedure :: is_initialized => dlr_is_initialized
    procedure :: clone => dlr_clone
    procedure :: assign => dlr_assign
    generic :: assignment(=) => assign  ! Overload assignment operator
    final :: dlr_finalize
  end type

