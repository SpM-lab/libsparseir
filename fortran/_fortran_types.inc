  type :: spir_basis_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_basis_release
  end type

  type :: spir_funcs_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_funcs_release
  end type

  type :: spir_kernel_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_kernel_release
  end type

  type :: spir_sampling_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_sampling_release
  end type

  type :: spir_sve_result_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_sve_result_release
  end type