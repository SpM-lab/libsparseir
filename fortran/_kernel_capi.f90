    ! Clone an existing kernel
    function c_spir_clone_kernel(src) bind(c, name='spir_clone_kernel')
      import c_ptr
      type(c_ptr), value :: src
      type(c_ptr) :: c_spir_clone_kernel
    end function

    ! Check if kernel's shared_ptr is assigned
    function c_spir_is_assigned_kernel(k) bind(c, name='spir_is_assigned_kernel')
      import c_ptr, c_int
      type(c_ptr), value :: k
      integer(c_int) :: c_spir_is_assigned_kernel
    end function

    ! Destroy a kernel
    subroutine c_spir_destroy_kernel(k) bind(c, name='spir_destroy_kernel')
      import c_ptr
      type(c_ptr), value :: k
    end subroutine
    

    ! Create a new logistic kernel
    function c_spir_logistic_kernel_new(lambda) bind(c, name='spir_logistic_kernel_new')
      import c_ptr, c_double
      real(c_double), value :: lambda
      type(c_ptr) :: c_spir_logistic_kernel_new
    end function

    ! Get the domain of a kernel
    function c_spir_kernel_domain(k, xmin, xmax, ymin, ymax) bind(c, name='spir_kernel_domain')
      import c_ptr, c_double, c_int
      type(c_ptr), value :: k
      real(c_double) :: xmin, xmax, ymin, ymax
      integer(c_int) :: c_spir_kernel_domain
    end function

