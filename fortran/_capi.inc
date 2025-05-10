  ! Clone an existing kernel
  function c_spir_clone_kernel(src) &
      bind(c, name='spir_clone_kernel')
    import c_ptr
    type(c_ptr), value :: src
    type(c_ptr) :: c_spir_clone_kernel
  end function

  ! Check if kernel's shared_ptr is assigned
  function c_spir_is_assigned_kernel(k) &
      bind(c, name='spir_is_assigned_kernel')
    import c_ptr, c_int
    type(c_ptr), value :: k
    integer(c_int) :: c_spir_is_assigned_kernel
  end function

  ! Destroy a kernel
  subroutine c_spir_destroy_kernel(k) &
      bind(c, name='spir_destroy_kernel')
    import c_ptr
    type(c_ptr), value :: k
  end subroutine

  ! Clone an existing sve_result
  function c_spir_clone_sve_result(src) &
      bind(c, name='spir_clone_sve_result')
    import c_ptr
    type(c_ptr), value :: src
    type(c_ptr) :: c_spir_clone_sve_result
  end function

  ! Check if sve_result's shared_ptr is assigned
  function c_spir_is_assigned_sve_result(k) &
      bind(c, name='spir_is_assigned_sve_result')
    import c_ptr, c_int
    type(c_ptr), value :: k
    integer(c_int) :: c_spir_is_assigned_sve_result
  end function

  ! Destroy a sve_result
  subroutine c_spir_destroy_sve_result(k) &
      bind(c, name='spir_destroy_sve_result')
    import c_ptr
    type(c_ptr), value :: k
  end subroutine

  ! Clone an existing funcs
  function c_spir_clone_funcs(src) &
      bind(c, name='spir_clone_funcs')
    import c_ptr
    type(c_ptr), value :: src
    type(c_ptr) :: c_spir_clone_funcs
  end function

  ! Check if funcs's shared_ptr is assigned
  function c_spir_is_assigned_funcs(k) &
      bind(c, name='spir_is_assigned_funcs')
    import c_ptr, c_int
    type(c_ptr), value :: k
    integer(c_int) :: c_spir_is_assigned_funcs
  end function

  ! Destroy a funcs
  subroutine c_spir_destroy_funcs(k) &
      bind(c, name='spir_destroy_funcs')
    import c_ptr
    type(c_ptr), value :: k
  end subroutine

  ! Clone an existing matsubara_funcs
  function c_spir_clone_matsubara_funcs(src) &
      bind(c, name='spir_clone_matsubara_funcs')
    import c_ptr
    type(c_ptr), value :: src
    type(c_ptr) :: c_spir_clone_matsubara_funcs
  end function

  ! Check if matsubara_funcs's shared_ptr is assigned
  function c_spir_is_assigned_matsubara_funcs(k) &
      bind(c, name='spir_is_assigned_matsubara_funcs')
    import c_ptr, c_int
    type(c_ptr), value :: k
    integer(c_int) :: c_spir_is_assigned_matsubara_funcs
  end function

  ! Destroy a matsubara_funcs
  subroutine c_spir_destroy_matsubara_funcs(k) &
      bind(c, name='spir_destroy_matsubara_funcs')
    import c_ptr
    type(c_ptr), value :: k
  end subroutine

  ! Clone an existing finite_temp_basis
  function c_spir_clone_finite_temp_basis(src) &
      bind(c, name='spir_clone_finite_temp_basis')
    import c_ptr
    type(c_ptr), value :: src
    type(c_ptr) :: c_spir_clone_finite_temp_basis
  end function

  ! Check if finite_temp_basis's shared_ptr is assigned
  function c_spir_is_assigned_finite_temp_basis(k) &
      bind(c, name='spir_is_assigned_finite_temp_basis')
    import c_ptr, c_int
    type(c_ptr), value :: k
    integer(c_int) :: c_spir_is_assigned_finite_temp_basis
  end function

  ! Destroy a finite_temp_basis
  subroutine c_spir_destroy_finite_temp_basis(k) &
      bind(c, name='spir_destroy_finite_temp_basis')
    import c_ptr
    type(c_ptr), value :: k
  end subroutine

  ! Clone an existing sampling
  function c_spir_clone_sampling(src) &
      bind(c, name='spir_clone_sampling')
    import c_ptr
    type(c_ptr), value :: src
    type(c_ptr) :: c_spir_clone_sampling
  end function

  ! Check if sampling's shared_ptr is assigned
  function c_spir_is_assigned_sampling(k) &
      bind(c, name='spir_is_assigned_sampling')
    import c_ptr, c_int
    type(c_ptr), value :: k
    integer(c_int) :: c_spir_is_assigned_sampling
  end function

  ! Destroy a sampling
  subroutine c_spir_destroy_sampling(k) &
      bind(c, name='spir_destroy_sampling')
    import c_ptr
    type(c_ptr), value :: k
  end subroutine

  ! Clone an existing dlr
  function c_spir_clone_dlr(src) &
      bind(c, name='spir_clone_dlr')
    import c_ptr
    type(c_ptr), value :: src
    type(c_ptr) :: c_spir_clone_dlr
  end function

  ! Check if dlr's shared_ptr is assigned
  function c_spir_is_assigned_dlr(k) &
      bind(c, name='spir_is_assigned_dlr')
    import c_ptr, c_int
    type(c_ptr), value :: k
    integer(c_int) :: c_spir_is_assigned_dlr
  end function

  ! Destroy a dlr
  subroutine c_spir_destroy_dlr(k) &
      bind(c, name='spir_destroy_dlr')
    import c_ptr
    type(c_ptr), value :: k
  end subroutine

