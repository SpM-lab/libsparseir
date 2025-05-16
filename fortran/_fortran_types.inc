  type :: spir_dlr_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_release_spir_dlr
  end type

  type :: spir_finite_temp_basis_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_release_spir_finite_temp_basis
  end type

  type :: spir_funcs_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_release_spir_funcs
  end type

  type :: spir_kernel_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_release_spir_kernel
  end type

  type :: spir_matsubara_funcs_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_release_spir_matsubara_funcs
  end type

  type :: spir_sampling_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_release_spir_sampling
  end type

  type :: spir_sve_result_handle
    type(c_ptr) :: handle = c_null_ptr
  contains
    final :: spir_release_spir_sve_result
  end type