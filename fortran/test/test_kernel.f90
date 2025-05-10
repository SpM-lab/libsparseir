! Simple test program for SparseIR Fortran bindings
program test_simple
  use sparseir
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double) :: xmin, xmax, ymin, ymax
  integer(c_int) :: stat

  type(kernel) :: k
  type(finite_temp_basis) :: basis
  double precision :: lambda, epsilon, beta, omega_max
  type(sve_result) :: sve

  ! Create a logistic kernel with lambda = 10.0
  lambda = 10.0d0
  print *, "Creating kernel with lambda =", lambda
  k = spir_logistic_kernel_new(lambda)
  
  print *, "Returned kernel ptr =", k%ptr
  print *, "Is ptr associated (C level)? ", c_associated(k%ptr)
  print * , "Checking is_assigned"
  print *, "Is object assigned (C++ level)? ", is_assigned(k)
  
  stat = spir_kernel_domain(k, xmin, xmax, ymin, ymax)
  print *, "Domain: [", xmin, ",", xmax, "] x [", ymin, ",", ymax, "]"
  print *, "Test completed successfully"

  ! Create a SVE result with epsilon = 1e-10
  epsilon = 1e-10
  sve = spir_sve_result_new(k, epsilon)

  ! Create a finite temperature basis with beta = 1.0, omega_max = 10.0, epsilon = 1e-10
  !beta = 1.0
  !omega_max = 10.0
  !basis = spir_finite_temp_basis_new(1, beta, omega_max, epsilon)

end program test_simple 