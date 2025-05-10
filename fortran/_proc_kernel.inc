    ! Wrapper for kernel creation
    module function spir_logistic_kernel_new(lambda) result(result)
      double precision, intent(in) :: lambda
      type(kernel) :: result
    end function

    ! Wrapper for kernel domain
    module function spir_kernel_domain(k, xmin, xmax, ymin, ymax) result(stat)
      type(kernel), intent(in) :: k
      real(c_double), intent(out) :: xmin, xmax, ymin, ymax
      integer(c_int) :: stat
    end function