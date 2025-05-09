    ! Check if kernel is initialized
    module function kernel_is_initialized(this) result(initialized)
      class(spir_kernel), intent(in) :: this
      logical :: initialized
    end function

    ! Clone the kernel object (create a copy)
    module function kernel_clone(this) result(copy)
      class(spir_kernel), intent(in) :: this
      type(spir_kernel) :: copy
    end function

    ! Assignment operator implementation
    module subroutine kernel_assign(lhs, rhs)
      class(spir_kernel), intent(inout) :: lhs
      class(spir_kernel), intent(in) :: rhs
    end subroutine

    ! Finalizer for kernel
    module subroutine kernel_finalize(this)
      type(spir_kernel), intent(inout) :: this
    end subroutine

    ! Check if kernel's shared_ptr is assigned (contains a valid C++ object)
    module function is_assigned(kernel) result(assigned)
      type(spir_kernel), intent(in) :: kernel
      logical :: assigned
    end function

    ! Wrapper for kernel creation
    module function spir_logistic_kernel_new(lambda) result(kernel)
      real(c_double), intent(in) :: lambda
      type(spir_kernel) :: kernel
    end function

    ! Wrapper for kernel domain
    module function spir_kernel_domain(kernel, xmin, xmax, ymin, ymax) result(stat)
      type(spir_kernel), intent(in) :: kernel
      real(c_double), intent(out) :: xmin, xmax, ymin, ymax
      integer(c_int) :: stat
    end function