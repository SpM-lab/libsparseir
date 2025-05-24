! Simple test program for SparseIR Fortran bindings
program test_kernel
  use sparseir
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double),target :: xmin, xmax, ymin, ymax
  integer(c_int), target :: status
  integer(c_int), target :: ntaus, nmatsus
  integer(c_int), target :: npoles

  real(c_double) :: beta = 10.0_c_double
  real(c_double) :: omega_max = 2.0_c_double
  real(c_double) :: epsilon = 1.0e-10_c_double
  real(c_double) :: lambda
  integer(c_int) :: positive_only = 1
  integer(c_int) :: order = SPIR_ORDER_COLUMN_MAJOR

  type(c_ptr) :: k_ptr, k_copy_ptr
  type(c_ptr) :: sve_ptr
  type(c_ptr) :: basis_ptr, dlr_ptr
  type(c_ptr) :: tau_sampling_ptr, matsu_sampling_ptr
  integer(c_int), target :: basis_size
  real(c_double), allocatable, target :: taus(:)
  integer(c_int64_t), allocatable, target :: matsus(:) ! Use int64_t for Matsubara indices
  real(c_double), allocatable, target :: poles(:)

  ! Parameters
  !beta = 10.0_c_double
  !omega_max = 2.0_c_double
  !epsilon = 1.0e-10_c_double
  lambda = beta * omega_max

  ! Create a new kernel
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

  ! Tau sampling
  print *, "Sampling tau"
  status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntaus))
  if (status /= 0) then
    print *, "Error getting number of tau points"
    stop
  end if
  print *, "Number of tau points =", ntaus
  allocate(taus(ntaus))
  status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))
  if (status /= 0) then
    print *, "Error getting tau points"
    stop
  end if
  print *, "Tau =", taus
  tau_sampling_ptr = c_spir_tau_sampling_new( &
    basis_ptr, ntaus, c_loc(taus), c_loc(status))
  if (status /= 0) then
    print *, "Error sampling tau"
    stop
  end if

  ! Matsubara sampling
  print *, "Sampling matsubara"
  status = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only, c_loc(nmatsus))
  if (status /= 0) then
    print *, "Error getting number of matsubara points"
    stop
  end if
  print *, "Number of matsubara points =", nmatsus
  allocate(matsus(nmatsus))
  status = c_spir_basis_get_default_matsus(basis_ptr, positive_only, c_loc(matsus))
  if (status /= 0) then
    print *, "Error getting matsubara points"
    stop
  end if
  print *, "Matsubara =", matsus
  matsu_sampling_ptr = c_spir_matsu_sampling_new( &
    basis_ptr, positive_only, nmatsus, c_loc(matsus), c_loc(status))
  if (status /= 0) then
    print *, "Error sampling matsubara"
    stop
  end if

  ! DLR
  print *, "Creating DLR"
  dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
  if (status /= 0) then
    print *, "Error creating DLR"
    stop
  end if
  if (.not. c_associated(dlr_ptr)) then
    print *, "Error: DLR is not assigned"
    stop
  end if

  ! Get the number of poles
  print *, "Getting number of poles"
  status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))
  if (status /= 0) then
    print *, "Error getting number of poles"
    stop
  end if
  print *, "Number of poles =", npoles

  ! Get the poles
  print *, "Getting poles"
  allocate(poles(npoles))
  status = c_spir_dlr_get_poles(dlr_ptr, c_loc(poles))
  if (status /= 0) then
    print *, "Error getting poles"
    stop
  end if
  print *, "Poles =", poles

  ! Deallocate
  deallocate(taus)
  deallocate(matsus)
  deallocate(poles)

  ! Release
  call c_spir_kernel_release(k_ptr)
  call c_spir_kernel_release(k_copy_ptr)
  call c_spir_sve_result_release(sve_ptr)
  call c_spir_basis_release(basis_ptr)
  call c_spir_basis_release(dlr_ptr)
  call c_spir_sampling_release(tau_sampling_ptr)
  call c_spir_sampling_release(matsu_sampling_ptr)

end program test_kernel
