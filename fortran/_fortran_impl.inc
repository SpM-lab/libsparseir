  ! Assignment operator implementation
  subroutine assign_spir_basis(lhs, rhs)
    type(spir_basis_handle), intent(inout) :: lhs
    type(spir_basis_handle), intent(in) :: rhs
    
    ! Check for self-assignment
    if (c_associated(lhs%handle, rhs%handle)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%handle)) then
      call c_spir_basis_release(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_basis_clone(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_basis
  subroutine spir_basis_release(this)
    type(spir_basis_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_basis_release(this%handle)
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
      call c_spir_funcs_release(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_funcs_clone(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_funcs
  subroutine spir_funcs_release(this)
    type(spir_funcs_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_funcs_release(this%handle)
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
      call c_spir_kernel_release(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_kernel_clone(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_kernel
  subroutine spir_kernel_release(this)
    type(spir_kernel_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_kernel_release(this%handle)
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
      call c_spir_sampling_release(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_sampling_clone(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_sampling
  subroutine spir_sampling_release(this)
    type(spir_sampling_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_sampling_release(this%handle)
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
      call c_spir_sve_result_release(lhs%handle)
      lhs%handle = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%handle)) then
      lhs%handle = c_spir_sve_result_clone(rhs%handle)
    end if
  end subroutine

  ! Finalizer for spir_sve_result
  subroutine spir_sve_result_release(this)
    type(spir_sve_result_handle), intent(inout) :: this
    
    if (c_associated(this%handle)) then
      call c_spir_sve_result_release(this%handle)
      this%handle = c_null_ptr
    end if
  end subroutine