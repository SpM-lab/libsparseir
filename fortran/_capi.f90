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

