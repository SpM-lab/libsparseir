! Test program for SparseIR Fortran bindings
program test_kernel
  use sparseir
  use, intrinsic :: iso_c_binding
  implicit none

  type(spir_kernel) :: kernel
  real(c_double) :: lambda, xmin, xmax, ymin, ymax
  integer :: stat
  
  ! Create a logistic kernel with lambda = 10.0
  lambda = 10.0d0
  kernel = spir_logistic_kernel_new(lambda)
  
  if (.not. kernel%is_initialized()) then
    print *, "Failed to create kernel"
    stop 1
  end if
  
  ! Get the domain of the kernel
  stat = spir_kernel_domain(kernel, xmin, xmax, ymin, ymax)
  
  if (stat /= 0) then
    print *, "Failed to get kernel domain, status =", stat
    call spir_destroy_kernel(kernel)
    stop 1
  end if
  
  print *, "Kernel domain:"
  print *, "  x: [", xmin, ", ", xmax, "]"
  print *, "  y: [", ymin, ", ", ymax, "]"
  
  ! Clean up
  call spir_destroy_kernel(kernel)
  
  print *, "Test completed successfully"
  
end program test_kernel 