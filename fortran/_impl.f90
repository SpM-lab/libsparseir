  ! Check if kernel is initialized
  module procedure kernel_is_initialized
    initialized = c_associated(this%ptr)
  end procedure

  ! Clone the kernel object (create a copy)
  module procedure kernel_clone
    if (.not. c_associated(this%ptr)) then
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the kernel
    copy%ptr = c_spir_clone_kernel(this%ptr)
  end procedure

  ! Assignment operator implementation
  module procedure kernel_assign
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      call c_spir_destroy_kernel(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      lhs%ptr = c_spir_clone_kernel(rhs%ptr)
    end if
  end procedure

  ! Finalizer for kernel
  module procedure kernel_finalize
    if (c_associated(this%ptr)) then
      call c_spir_destroy_kernel(this%ptr)
      this%ptr = c_null_ptr
    end if
  end procedure

  ! Check if kernel's shared_ptr is assigned (contains a valid C++ object)
  module procedure kernel_is_assigned
    if (.not. c_associated(kernel%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_kernel(kernel%ptr) /= 0)
  end procedure

  ! Check if sve_result is initialized
  module procedure sve_result_is_initialized
    initialized = c_associated(this%ptr)
  end procedure

  ! Clone the sve_result object (create a copy)
  module procedure sve_result_clone
    if (.not. c_associated(this%ptr)) then
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the sve_result
    copy%ptr = c_spir_clone_sve_result(this%ptr)
  end procedure

  ! Assignment operator implementation
  module procedure sve_result_assign
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      call c_spir_destroy_sve_result(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      lhs%ptr = c_spir_clone_sve_result(rhs%ptr)
    end if
  end procedure

  ! Finalizer for sve_result
  module procedure sve_result_finalize
    if (c_associated(this%ptr)) then
      call c_spir_destroy_sve_result(this%ptr)
      this%ptr = c_null_ptr
    end if
  end procedure

  ! Check if sve_result's shared_ptr is assigned (contains a valid C++ object)
  module procedure sve_result_is_assigned
    if (.not. c_associated(sve_result%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_sve_result(sve_result%ptr) /= 0)
  end procedure

