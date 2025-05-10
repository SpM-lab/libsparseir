    
    function c_spir_finite_temp_basis_new(b, statistics, beta, omega_max, epsilon) &
        bind(c, name='spir_finite_temp_basis_new') result(status)
      import c_ptr, c_double, c_int
      type(c_ptr), intent(out) :: b
      integer(c_int), value :: statistics
      real(c_double), value :: beta
      real(c_double), value :: omega_max
      real(c_double), value :: epsilon
      integer(c_int) :: status
    end function
    
    function c_spir_finite_temp_basis_get_size(b, size) &
        bind(c, name='spir_finite_temp_basis_get_size') result(status)
      import c_ptr, c_int
      type(c_ptr), value :: b
      integer(c_int), intent(out) :: size
      integer(c_int) :: status
    end function