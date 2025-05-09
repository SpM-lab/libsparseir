    ! Create a new logistic kernel
    function c_spir_logistic_kernel_new(kernel, lambda) bind(c, name='spir_logistic_kernel_new')
      import c_ptr, c_double, c_int
      type(c_ptr), intent(out) :: kernel
      real(c_double), value :: lambda
      integer(c_int) :: c_spir_logistic_kernel_new
    end function

    ! Get the domain of a kernel
    function c_spir_kernel_domain(k, xmin, xmax, ymin, ymax) bind(c, name='spir_kernel_domain')
      import c_ptr, c_double, c_int
      type(c_ptr), value :: k
      real(c_double) :: xmin, xmax, ymin, ymax
      integer(c_int) :: c_spir_kernel_domain
    end function

