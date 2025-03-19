! Simple test program for SparseIR Fortran bindings
program test_simple
  use sparseir
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double) :: xmin, xmax, ymin, ymax
  integer(c_int) :: stat

  type(spir_kernel) :: kernel
  real(c_double) :: lambda
  
  ! Create a logistic kernel with lambda = 10.0
  lambda = 10.0d0
  print *, "Creating kernel with lambda =", lambda
  kernel = spir_logistic_kernel_new(lambda)
  
  print *, "Returned kernel ptr =", kernel%ptr
  print *, "Is ptr associated (C level)? ", c_associated(kernel%ptr)
  print * , "Checking is_assigned"
  print *, "Is object assigned (C++ level)? ", is_assigned(kernel)
  
  stat = spir_kernel_domain(kernel, xmin, xmax, ymin, ymax)
  print *, "Domain: [", xmin, ",", xmax, "] x [", ymin, ",", ymax, "]"
  print *, "Test completed successfully"
  
end program test_simple 