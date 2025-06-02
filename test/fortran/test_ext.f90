! Simple test program for SparseIR Fortran bindings
program test_ext
   use sparseir
   use sparseir_ext
   implicit none

   type(IR) :: irobj
   real(DP), parameter :: beta = 10.0_DP
   real(DP), parameter :: omega_max = 2.0_DP
   real(DP), parameter :: epsilon = 1.0e-10_DP
   real(DP), parameter :: lambda = beta * omega_max
   logical, parameter :: positive_only = .false.

   ! Test Fermionic case
   call test_case(irobj, "Fermionic")

contains
   subroutine test_case(obj, case_name)
      type(IR), intent(inout) :: obj
      character(len=*), intent(in) :: case_name

      real(DP), allocatable :: coeffs(:,:), g_ir(:,:)
      complex(DP), allocatable :: giw(:,:), giw_reconst(:,:), g_ir2_z(:,:), gtau_z(:,:)
      integer :: i, j
      real(DP) :: r

      print *, "Testing ", case_name, " case"

      ! Initialize IR object
      call init_ir(obj, beta, lambda, epsilon, positive_only)

      ! Allocate arrays
      allocate(coeffs(obj%size, 1))
      allocate(g_ir(obj%size, 1))
      allocate(g_ir2_z(obj%size, 1))
      allocate(gtau_z(obj%ntau, 1))
      allocate(giw(obj%nfreq_f, 1))
      allocate(giw_reconst(obj%nfreq_f, 1))

      ! Generate random coefficients
      do i = 1, obj%size
         do j = 1, 1
            call random_number(r)
            coeffs(i,j) = (2.0_DP * r - 1.0_DP) * sqrt(abs(obj%s(i)))
         end do
      end do

      ! Convert DLR coefficients to IR coefficients
      g_ir = coeffs

      ! Evaluate Green's function at Matsubara frequencies from IR
      call evaluate_matsubara_f_zz(obj, 1, g_ir, giw)

      ! Convert Matsubara frequencies back to IR
      call fit_matsubara_f_zz(obj, 1, giw, g_ir2_z)

      ! Compare IR coefficients (using real part of g_ir2_z)
      if (.not. compare_with_relative_error_d(g_ir, real(g_ir2_z), 10.0_DP * epsilon)) then
         print *, "Error: IR coefficients do not match after transformation cycle"
         stop
      end if

      ! Evaluate Green's function at tau points
      call evaluate_tau_zz(obj, 1, g_ir2_z, gtau_z)

      ! Convert tau points back to IR
      call fit_tau_zz(obj%tau_smpl_ptr, 1, gtau_z, g_ir2_z)

      ! Evaluate Green's function at Matsubara frequencies again
      call evaluate_matsubara_f_zz(obj, 1, g_ir2_z, giw_reconst)

      ! Compare the original and reconstructed Matsubara frequencies
      if (.not. compare_with_relative_error_z(giw, giw_reconst, 10.0_DP * epsilon)) then
         print *, "Error: Matsubara frequencies do not match after transformation cycle"
         stop
      end if

      ! Deallocate arrays
      deallocate(coeffs)
      deallocate(g_ir)
      deallocate(g_ir2_z)
      deallocate(gtau_z)
      deallocate(giw)
      deallocate(giw_reconst)

      ! Finalize IR object
      call finalize_ir(obj)
   end subroutine test_case

   function compare_with_relative_error_d(a, b, tol) result(is_close)
      real(DP), intent(in) :: a(:,:), b(:,:)
      real(DP), intent(in) :: tol
      logical :: is_close
      real(DP) :: max_diff, max_ref
      integer :: i, j

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do j = 1, size(a, 2)
         do i = 1, size(a, 1)
            max_diff = max(max_diff, abs(a(i,j) - b(i,j)))
            max_ref = max(max_ref, abs(a(i,j)))
         end do
      end do

      is_close = max_diff <= tol * max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff / max_ref
      end if
   end function compare_with_relative_error_d

   function compare_with_relative_error_z(a, b, tol) result(is_close)
      complex(DP), intent(in) :: a(:,:), b(:,:)
      real(DP), intent(in) :: tol
      logical :: is_close
      real(DP) :: max_diff, max_ref
      integer :: i, j

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do j = 1, size(a, 2)
         do i = 1, size(a, 1)
            max_diff = max(max_diff, abs(a(i,j) - b(i,j)))
            max_ref = max(max_ref, abs(a(i,j)))
         end do
      end do

      is_close = max_diff <= tol * max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff / max_ref
      end if
   end function compare_with_relative_error_z
end program test_ext