  ! Check if kernel is initialized
  module procedure kernel_is_initialized
    initialized = c_associated(this%ptr)
  end procedure

  ! Clone the kernel object (create a copy)
  module procedure kernel_clone
    print *, "Fortran: Cloning kernel with ptr =", this%ptr
    
    if (.not. c_associated(this%ptr)) then
      print *, "Fortran: Source ptr is null, returning uninitialized copy"
      copy%ptr = c_null_ptr
      return
    end if
    
    ! Call C function to clone the kernel
    copy%ptr = c_spir_clone_kernel(this%ptr)
    print *, "Fortran: Cloned kernel, new ptr =", copy%ptr
    
    if (.not. c_associated(copy%ptr)) then
      print *, "Fortran: Warning - Clone operation returned NULL pointer"
    end if
  end procedure

  ! Assignment operator implementation
  module procedure kernel_assign
    print *, "Fortran: Assignment operator called"
    print *, "  LHS ptr =", lhs%ptr
    print *, "  RHS ptr =", rhs%ptr
    
    ! Check for self-assignment
    if (c_associated(lhs%ptr, rhs%ptr)) then
      print *, "Fortran: Self-assignment detected, nothing to do"
      return
    end if
    
    ! Clean up existing resource if present
    if (c_associated(lhs%ptr)) then
      print *, "Fortran: Cleaning up existing LHS resource"
      call c_spir_destroy_kernel(lhs%ptr)
      lhs%ptr = c_null_ptr
    end if
    
    ! If RHS is valid, clone it
    if (c_associated(rhs%ptr)) then
      print *, "Fortran: Cloning RHS to LHS"
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
  module procedure is_assigned
    !print *, "Fortran: Checking if kernel's shared_ptr is assigned"
    
    if (.not. c_associated(kernel%ptr)) then
      !print *, "Fortran: Pointer is not associated at C level"
      assigned = .false.
      return
    end if
    
    assigned = (c_spir_is_assigned_kernel(kernel%ptr) /= 0)
    !print *, "Fortran: C++ shared_ptr is_assigned =", assigned
  end procedure

  ! Wrapper for kernel creation
  module procedure spir_logistic_kernel_new
    integer(c_int) :: status
    real(c_double) :: lambda_c 

    print *, "Calling C function to create logistic kernel with lambda =", lambda
    lambda_c = real(lambda, c_double)
    status = c_spir_logistic_kernel_new(kernel%ptr, lambda_c)
    print *, "C function returned kernel ptr =", kernel%ptr
    
    if (status /= 0) then
      print *, "Warning: C function returned error status =", status
      kernel%ptr = c_null_ptr
    end if
  end procedure

  ! Wrapper for kernel domain
  module procedure spir_kernel_domain
    if (.not. kernel%is_initialized()) then
      print *, "Error: Kernel not initialized in spir_kernel_domain"
      stat = -1
      return
    end if
    
    print *, "Calling C function c_spir_kernel_domain with ptr =", kernel%ptr
    print *, "Is pointer associated? ", c_associated(kernel%ptr)
    
    ! Initialize output variables
    xmin = 0.0_c_double
    xmax = 0.0_c_double
    ymin = 0.0_c_double
    ymax = 0.0_c_double
    
    stat = c_spir_kernel_domain(kernel%ptr, xmin, xmax, ymin, ymax)
    print *, "C function c_spir_kernel_domain returned status =", stat
  end procedure