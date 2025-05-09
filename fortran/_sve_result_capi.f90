    function c_spir_sve_result_new(sve, k, epsilon) &
        bind(c, name='spir_sve_result_new') result(status)
      import c_ptr, c_double, c_int
      type(c_ptr), intent(out) :: sve
      type(c_ptr), value :: k
      real(c_double), value :: epsilon
      integer(c_int) :: status
    end function