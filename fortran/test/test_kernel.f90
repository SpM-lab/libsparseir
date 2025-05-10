! Simple test program for SparseIR Fortran bindings
program test_kernel
  use sparseir
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double),target :: xmin, xmax, ymin, ymax
  integer(c_int) :: status, is_assigned

  real(c_double) :: beta = 1.0_c_double
  real(c_double) :: omega_max = 10.0_c_double
  real(c_double) :: epsilon = 1.0e-6_c_double
  real(c_double) :: lambda
  type(spir_kernel_handle) :: k
  type(spir_finite_temp_basis_handle) :: basis
  type(spir_sve_result_handle) :: sve
  integer(c_int), target :: basis_size

  ! Create a new kernel
  lambda = beta * omega_max
  status = c_spir_logistic_kernel_new(k%handle, lambda)
  if (status /= 0) then
    print *, "Error creating kernel"
    stop
  end if
  is_assigned = c_spir_is_assigned_kernel(k%handle)
  if (is_assigned /= 1) then
    print *, "Error: kernel is not assigned"
    stop
  end if

  status = c_spir_kernel_domain(k%handle, c_loc(xmin), c_loc(xmax), c_loc(ymin), c_loc(ymax))
  if (status /= 0) then
    print *, "Error: kernel domain is not assigned"
    stop
  end if
  print *, "Kernel domain =", xmin, xmax, ymin, ymax

  ! Create a new SVE result
  print *, "Creating SVE result"
  status = c_spir_sve_result_new(sve%handle, k%handle, epsilon)
  if (status /= 0) then
    print *, "Error creating SVE result"
    stop
  end if
  print *, "Returned SVE result ptr =", sve%handle
  is_assigned = c_spir_is_assigned_sve_result(sve%handle)
  if (is_assigned /= 1) then
    print *, "Error: SVE result is not assigned"
    stop
  end if


  ! Create a finite temperature basis with beta = 1.0, omega_max = 10.0, epsilon = 1e-10
  print *, "Creating finite temperature basis"
  status = c_spir_finite_temp_basis_new(basis%handle, SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon)
  if (status /= 0) then
    print *, "Error creating finite temperature basis"
    stop
  end if
  status = c_spir_is_assigned_kernel(basis%handle)
  if (status /= 1) then
    print *, "Error: basis is not assigned"
    stop
  end if

  ! Get the size of the basis
  print *, "Getting basis size"
  status = c_spir_finite_temp_basis_get_size(basis%handle, c_loc(basis_size))
  if (status /= 0) then
    print *, "Error getting basis size"
    stop
  end if
  print *, "Basis size =", basis_size

  ! Create a finite temperature basis with beta = 1.0, omega_max = 10.0, epsilon = 1e-10
  !basis%handle = spir_finite_temp_basis_new(1, beta, omega_max, epsilon)

end program test_kernel 