module procedure spir_finite_temp_basis_new
  integer(c_int) :: status

  integer(c_int) :: c_statistics
  real(c_double) :: c_beta
  real(c_double) :: c_omega_max
  real(c_double) :: c_epsilon

  c_statistics = statistics
  c_beta = beta
  c_omega_max = omega_max
  c_epsilon = epsilon

  status = c_spir_finite_temp_basis_new(result%ptr, c_statistics, c_beta, c_omega_max, c_epsilon)

  if (status /= 0) then
    print *, "Warning: C function returned error status =", status
    result%ptr = c_null_ptr
  end if

end procedure