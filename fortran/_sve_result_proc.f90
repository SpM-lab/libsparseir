    module function spir_sve_result_new(k, epsilon) result(result)
      type(spir_kernel), intent(in) :: k
      real(c_double), intent(in) :: epsilon
      type(spir_sve_result) :: result
    end function
