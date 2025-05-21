! Simple test program for SparseIR Fortran bindings
program test_kernel
  use sparseir
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double),target :: xmin, xmax, ymin, ymax
  integer(c_int32_t), target :: status

  real(c_double) :: beta = 1.0_c_double
  real(c_double) :: omega_max = 10.0_c_double
  real(c_double) :: epsilon = 1.0e-6_c_double
  real(c_double) :: lambda
  type(c_ptr) :: k_ptr, k_copy_ptr
  type(c_ptr) :: sve_ptr
  type(c_ptr) :: basis_ptr
  integer(c_int32_t), target :: basis_size

  ! Create a new kernel
  lambda = beta * omega_max
  k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
  if (status /= 0) then
    print *, "Error creating kernel"
    stop
  end if
  if (.not. c_associated(k_ptr)) then
    print *, "Error: kernel is not assigned"
    stop
  end if

  status = c_spir_kernel_domain(k_ptr, c_loc(xmin), c_loc(xmax), c_loc(ymin), c_loc(ymax))
  if (status /= 0) then
    print *, "Error: kernel domain is not assigned"
    stop
  end if
  print *, "Kernel domain =", xmin, xmax, ymin, ymax

  ! Create a copy of the kernel.
  ! All C objects are immutable, and can be cloned without copying the data.
  ! After releasing the original kernel, the copy is still valid.
  ! The same applies to all other C objects.
  k_copy_ptr = c_spir_kernel_clone(k_ptr)

  ! Create a new SVE result
  print *, "Creating SVE result"
  sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, c_loc(status))
  if (status /= 0) then
    print *, "Error creating SVE result"
    stop
  end if
  if (.not. c_associated(sve_ptr)) then
    print *, "Error: SVE result is not assigned"
    stop
  end if

  ! Create a finite temperature basis with beta = 1.0, omega_max = 10.0, epsilon = 1e-10
  print *, "Creating finite temperature basis"
  basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, k_ptr, sve_ptr, c_loc(status))
  if (status /= 0) then
    print *, "Error creating finite temperature basis"
    stop
  end if
  if (.not. c_associated(basis_ptr)) then
    print *, "Error: basis is not assigned"
    stop
  end if

  ! Get the size of the basis
  print *, "Getting basis size"
  status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))
  if (status /= 0) then
    print *, "Error getting basis size"
    stop
  end if
  print *, "Basis size =", basis_size

end program test_kernel 