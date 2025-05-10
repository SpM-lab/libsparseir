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