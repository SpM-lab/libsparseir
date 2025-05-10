    module function spir_finite_temp_basis_new(statistics, beta, omega_max, epsilon) result(result)
      integer, intent(in) :: statistics
      real(8), intent(in) :: beta
      real(8), intent(in) :: omega_max
      real(8), intent(in) :: epsilon
      type(spir_finite_temp_basis) :: result
    end function
