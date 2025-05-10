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
    if (.not. c_associated(this%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_kernel(this%ptr) /= 0)
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
    if (.not. c_associated(this%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_sve_result(this%ptr) /= 0)
  end procedure

  ! Check if funcs is initialized
  module procedure funcs_is_initialized
    initialized = c_associated(this%ptr)
  end procedure

  ! Clone the funcs object (create a copy)
  module procedure funcs_clone
    if (.not. c_associated(this%ptr)) then
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the funcs
    copy%ptr = c_spir_clone_funcs(this%ptr)
  end procedure

  ! Assignment operator implementation
  module procedure funcs_assign
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      call c_spir_destroy_funcs(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      lhs%ptr = c_spir_clone_funcs(rhs%ptr)
    end if
  end procedure

  ! Finalizer for funcs
  module procedure funcs_finalize
    if (c_associated(this%ptr)) then
      call c_spir_destroy_funcs(this%ptr)
      this%ptr = c_null_ptr
    end if
  end procedure

  ! Check if funcs's shared_ptr is assigned (contains a valid C++ object)
  module procedure funcs_is_assigned
    if (.not. c_associated(this%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_funcs(this%ptr) /= 0)
  end procedure

  ! Check if matsubara_funcs is initialized
  module procedure matsubara_funcs_is_initialized
    initialized = c_associated(this%ptr)
  end procedure

  ! Clone the matsubara_funcs object (create a copy)
  module procedure matsubara_funcs_clone
    if (.not. c_associated(this%ptr)) then
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the matsubara_funcs
    copy%ptr = c_spir_clone_matsubara_funcs(this%ptr)
  end procedure

  ! Assignment operator implementation
  module procedure matsubara_funcs_assign
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      call c_spir_destroy_matsubara_funcs(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      lhs%ptr = c_spir_clone_matsubara_funcs(rhs%ptr)
    end if
  end procedure

  ! Finalizer for matsubara_funcs
  module procedure matsubara_funcs_finalize
    if (c_associated(this%ptr)) then
      call c_spir_destroy_matsubara_funcs(this%ptr)
      this%ptr = c_null_ptr
    end if
  end procedure

  ! Check if matsubara_funcs's shared_ptr is assigned (contains a valid C++ object)
  module procedure matsubara_funcs_is_assigned
    if (.not. c_associated(this%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_matsubara_funcs(this%ptr) /= 0)
  end procedure

  ! Check if finite_temp_basis is initialized
  module procedure finite_temp_basis_is_initialized
    initialized = c_associated(this%ptr)
  end procedure

  ! Clone the finite_temp_basis object (create a copy)
  module procedure finite_temp_basis_clone
    if (.not. c_associated(this%ptr)) then
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the finite_temp_basis
    copy%ptr = c_spir_clone_finite_temp_basis(this%ptr)
  end procedure

  ! Assignment operator implementation
  module procedure finite_temp_basis_assign
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      call c_spir_destroy_finite_temp_basis(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      lhs%ptr = c_spir_clone_finite_temp_basis(rhs%ptr)
    end if
  end procedure

  ! Finalizer for finite_temp_basis
  module procedure finite_temp_basis_finalize
    if (c_associated(this%ptr)) then
      call c_spir_destroy_finite_temp_basis(this%ptr)
      this%ptr = c_null_ptr
    end if
  end procedure

  ! Check if finite_temp_basis's shared_ptr is assigned (contains a valid C++ object)
  module procedure finite_temp_basis_is_assigned
    if (.not. c_associated(this%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_finite_temp_basis(this%ptr) /= 0)
  end procedure

  ! Check if sampling is initialized
  module procedure sampling_is_initialized
    initialized = c_associated(this%ptr)
  end procedure

  ! Clone the sampling object (create a copy)
  module procedure sampling_clone
    if (.not. c_associated(this%ptr)) then
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the sampling
    copy%ptr = c_spir_clone_sampling(this%ptr)
  end procedure

  ! Assignment operator implementation
  module procedure sampling_assign
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      call c_spir_destroy_sampling(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      lhs%ptr = c_spir_clone_sampling(rhs%ptr)
    end if
  end procedure

  ! Finalizer for sampling
  module procedure sampling_finalize
    if (c_associated(this%ptr)) then
      call c_spir_destroy_sampling(this%ptr)
      this%ptr = c_null_ptr
    end if
  end procedure

  ! Check if sampling's shared_ptr is assigned (contains a valid C++ object)
  module procedure sampling_is_assigned
    if (.not. c_associated(this%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_sampling(this%ptr) /= 0)
  end procedure

  ! Check if dlr is initialized
  module procedure dlr_is_initialized
    initialized = c_associated(this%ptr)
  end procedure

  ! Clone the dlr object (create a copy)
  module procedure dlr_clone
    if (.not. c_associated(this%ptr)) then
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the dlr
    copy%ptr = c_spir_clone_dlr(this%ptr)
  end procedure

  ! Assignment operator implementation
  module procedure dlr_assign
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      call c_spir_destroy_dlr(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      lhs%ptr = c_spir_clone_dlr(rhs%ptr)
    end if
  end procedure

  ! Finalizer for dlr
  module procedure dlr_finalize
    if (c_associated(this%ptr)) then
      call c_spir_destroy_dlr(this%ptr)
      this%ptr = c_null_ptr
    end if
  end procedure

  ! Check if dlr's shared_ptr is assigned (contains a valid C++ object)
  module procedure dlr_is_assigned
    if (.not. c_associated(this%ptr)) then
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_dlr(this%ptr) /= 0)
  end procedure

