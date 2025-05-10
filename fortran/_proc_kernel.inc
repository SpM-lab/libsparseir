    ! Wrapper for kernel creation
    module function spir_logistic_kernel_new(lambda) result(kernel)
      double precision, intent(in) :: lambda
      type(spir_kernel) :: kernel
    end function

    ! Wrapper for kernel domain
    module function spir_kernel_domain(kernel, xmin, xmax, ymin, ymax) result(stat)
      type(spir_kernel), intent(in) :: kernel
      real(c_double), intent(out) :: xmin, xmax, ymin, ymax
      integer(c_int) :: stat
    end function