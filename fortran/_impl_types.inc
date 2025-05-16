  ! Assignment operator implementation
  subroutine assign_spir_dlr(lhs, rhs)
    type(spir_dlr_handle), intent(inout) :: lhs
    type(spir_dlr_handle), intent(in) :: rhs
    
    ! Check for self-assignment
    if (c_associated(lhs%handle, rhs%handle)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%handle)) then
      call c_spir_release_dlr(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_clone_dlr(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_dlr
  subroutine spir_release_spir_dlr(this)
    type(spir_dlr_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_release_dlr(this%handle)
      this%handle = c_null_ptr
    end if
  end subroutine

  ! Assignment operator implementation
  subroutine assign_spir_finite_temp_basis(lhs, rhs)
    type(spir_finite_temp_basis_handle), intent(inout) :: lhs
    type(spir_finite_temp_basis_handle), intent(in) :: rhs
    
    ! Check for self-assignment
    if (c_associated(lhs%handle, rhs%handle)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%handle)) then
      call c_spir_release_finite_temp_basis(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_clone_finite_temp_basis(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_finite_temp_basis
  subroutine spir_release_spir_finite_temp_basis(this)
    type(spir_finite_temp_basis_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_release_finite_temp_basis(this%handle)
      this%handle = c_null_ptr
    end if
  end subroutine

  ! Assignment operator implementation
  subroutine assign_spir_funcs(lhs, rhs)
    type(spir_funcs_handle), intent(inout) :: lhs
    type(spir_funcs_handle), intent(in) :: rhs
    
    ! Check for self-assignment
    if (c_associated(lhs%handle, rhs%handle)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%handle)) then
      call c_spir_release_funcs(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_clone_funcs(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_funcs
  subroutine spir_release_spir_funcs(this)
    type(spir_funcs_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_release_funcs(this%handle)
      this%handle = c_null_ptr
    end if
  end subroutine

  ! Assignment operator implementation
  subroutine assign_spir_kernel(lhs, rhs)
    type(spir_kernel_handle), intent(inout) :: lhs
    type(spir_kernel_handle), intent(in) :: rhs
    
    ! Check for self-assignment
    if (c_associated(lhs%handle, rhs%handle)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%handle)) then
      call c_spir_release_kernel(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_clone_kernel(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_kernel
  subroutine spir_release_spir_kernel(this)
    type(spir_kernel_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_release_kernel(this%handle)
      this%handle = c_null_ptr
    end if
  end subroutine

  ! Assignment operator implementation
  subroutine assign_spir_matsubara_funcs(lhs, rhs)
    type(spir_matsubara_funcs_handle), intent(inout) :: lhs
    type(spir_matsubara_funcs_handle), intent(in) :: rhs
    
    ! Check for self-assignment
    if (c_associated(lhs%handle, rhs%handle)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%handle)) then
      call c_spir_release_matsubara_funcs(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_clone_matsubara_funcs(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_matsubara_funcs
  subroutine spir_release_spir_matsubara_funcs(this)
    type(spir_matsubara_funcs_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_release_matsubara_funcs(this%handle)
      this%handle = c_null_ptr
    end if
  end subroutine

  ! Assignment operator implementation
  subroutine assign_spir_sampling(lhs, rhs)
    type(spir_sampling_handle), intent(inout) :: lhs
    type(spir_sampling_handle), intent(in) :: rhs
    
    ! Check for self-assignment
    if (c_associated(lhs%handle, rhs%handle)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%handle)) then
      call c_spir_release_sampling(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_clone_sampling(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_sampling
  subroutine spir_release_spir_sampling(this)
    type(spir_sampling_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_release_sampling(this%handle)
      this%handle = c_null_ptr
    end if
  end subroutine

  ! Assignment operator implementation
  subroutine assign_spir_sve_result(lhs, rhs)
    type(spir_sve_result_handle), intent(inout) :: lhs
    type(spir_sve_result_handle), intent(in) :: rhs
    
    ! Check for self-assignment
    if (c_associated(lhs%handle, rhs%handle)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%handle)) then
      call c_spir_release_sve_result(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_clone_sve_result(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_sve_result
  subroutine spir_release_spir_sve_result(this)
    type(spir_sve_result_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_release_sve_result(this%handle)
      this%handle = c_null_ptr
    end if
  end subroutine