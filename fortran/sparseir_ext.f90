! Extends the SparseIR library with additional functionality
MODULE sparseir_ext
   USE, INTRINSIC :: iso_c_binding
   USE sparseir
   IMPLICIT NONE

   INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
   INTEGER, PARAMETER :: sgl = selected_real_kind(6,30)
   INTEGER, PARAMETER :: i4b = selected_int_kind(9)
   INTEGER, PARAMETER :: i8b = selected_int_kind(18)

   !-----------------------------------------------------------------------
   TYPE IR
      !-----------------------------------------------------------------------
      !!
      !! This contains all the IR-basis objects,
      !! such as sampling points, and basis functions
      !!
      !
      INTEGER :: size
      !! total number of IR basis functions (size of s)
      INTEGER :: ntau
      !! total number of sampling points of imaginary time
      INTEGER :: nfreq_f
      !! total number of sampling Matsubara freqs (Fermionic)
      INTEGER :: nfreq_b
      !! total number of sampling Matsubara freqs (Bosonic)
      INTEGER :: nomega
      !! total number of sampling points of real frequency
      REAL(KIND = DP) :: beta
      !! inverse temperature
      REAL(KIND = DP) :: lambda
      !! lambda = 10^{nlambda},
      !! which determines maximum sampling point of real frequency
      REAL(KIND = DP) :: wmax
      !! maximum real frequency: wmax = lambda / beta
      REAL(KIND = DP) :: eps
      !! eps = 10^{-ndigit}
      REAL(KIND = DP) :: eps_svd
      !! This is used in the SVD fitting.
      REAL(KIND = DP), ALLOCATABLE :: s(:)
      !! singular values
      REAL(KIND = DP), ALLOCATABLE :: tau(:)
      !! sampling points of imaginary time
      REAL(KIND = DP), ALLOCATABLE :: omega(:)
      !! sampling points of real frequency
      INTEGER(8), ALLOCATABLE :: freq_f(:)
      !! integer part of sampling Matsubara freqs (Fermion)
      INTEGER(8), ALLOCATABLE :: freq_b(:)
      !! integer part of sampling Matsubara freqs (Boson)
      LOGICAL :: positive_only
      !! if true, take the Matsubara frequencies
      !! only from the positive region

      TYPE(c_ptr) :: basis_f_ptr
      !! pointer to the fermionic basis
      TYPE(c_ptr) :: basis_b_ptr
      !! pointer to the bosonic basis
      TYPE(c_ptr) :: sve_ptr
      !! pointer to the SVE result
      TYPE(c_ptr) :: k_ptr
      !! pointer to the kernel
      TYPE(c_ptr) :: tau_smpl_ptr
      !! pointer to the tau sampling points
      TYPE(c_ptr) :: matsu_f_smpl_ptr
      !! pointer to the fermionic frequency sampling points
      TYPE(c_ptr) :: matsu_b_smpl_ptr
      !! pointer to the bosonic frequency sampling points
      !-----------------------------------------------------------------------
   END TYPE IR
   !-----------------------------------------------------------------------

   INTERFACE evaluate_tau
       MODULE PROCEDURE evaluate_tau_zz, evaluate_tau_dd
   END INTERFACE evaluate_tau

   INTERFACE evaluate_matsubara
       MODULE PROCEDURE evaluate_matsubara_zz, evaluate_matsubara_dz
   END INTERFACE evaluate_matsubara

   INTERFACE evaluate_matsubara_f
       MODULE PROCEDURE evaluate_matsubara_f_zz, evaluate_matsubara_f_dz
   END INTERFACE evaluate_matsubara_f

   INTERFACE evaluate_matsubara_b
       MODULE PROCEDURE evaluate_matsubara_b_zz, evaluate_matsubara_b_dz
   END INTERFACE evaluate_matsubara_b

   INTERFACE fit_tau
       MODULE PROCEDURE fit_tau_zz, fit_tau_dd
   END INTERFACE fit_tau

   INTERFACE fit_matsubara
       MODULE PROCEDURE fit_matsubara_zz
   END INTERFACE fit_matsubara

   INTERFACE fit_matsubara_f
       MODULE PROCEDURE fit_matsubara_f_zz
   END INTERFACE fit_matsubara_f

   INTERFACE fit_matsubara_b
       MODULE PROCEDURE fit_matsubara_b_zz
   END INTERFACE fit_matsubara_b

   !
   !INTERFACE fit_matsubara_f
   !MODULE PROCEDURE fit_matsubara_f_zz, fit_matsubara_f_zd
   !END INTERFACE fit_matsubara_f
   !!
   !INTERFACE fit_matsubara_b
   !MODULE PROCEDURE fit_matsubara_b_zz, fit_matsubara_b_zd
   !END INTERFACE fit_matsubara_b

contains
   FUNCTION create_logistic_kernel(lambda) result(k_ptr)
      REAL(KIND = DP), INTENT(IN) :: lambda
      REAL(c_double), target :: lambda_c
      INTEGER(c_int), target :: status_c
      TYPE(c_ptr) :: k_ptr
      lambda_c = lambda
      k_ptr = c_spir_logistic_kernel_new(lambda_c, c_loc(status_c))
   end function create_logistic_kernel

   FUNCTION create_sve_result(lambda, eps, k_ptr) result(sve_ptr)
      REAL(KIND = DP), INTENT(IN) :: lambda
      REAL(KIND = DP), INTENT(IN) :: eps

      REAL(c_double), target :: lambda_c, eps_c
      INTEGER(c_int), target :: status_c

      TYPE(c_ptr), INTENT(IN) :: k_ptr
      TYPE(c_ptr) :: sve_ptr

      lambda_c = lambda
      eps_c = eps

      sve_ptr = c_spir_sve_result_new(k_ptr, eps_c, c_loc(status_c))
      IF (status_c /= 0) THEN
         CALL errore('create_sve_result', 'Error creating SVE result', status_c)
      ENDIF
   end function create_sve_result

   FUNCTION create_basis(statistics, beta, wmax, k_ptr, sve_ptr) result(basis_ptr)
      INTEGER, INTENT(IN) :: statistics
      REAL(KIND = DP), INTENT(IN) :: beta
      REAL(KIND = DP), INTENT(IN) :: wmax
      TYPE(c_ptr), INTENT(IN) :: k_ptr, sve_ptr

      INTEGER(c_int), target :: status_c
      TYPE(c_ptr) :: basis_ptr
      INTEGER(c_int) :: statistics_c
      REAL(c_double) :: beta_c, wmax_c

      statistics_c = statistics
      beta_c = beta
      wmax_c = wmax

      basis_ptr = c_spir_basis_new(statistics_c, beta_c, wmax_c, k_ptr, sve_ptr, c_loc(status_c))
      IF (status_c /= 0) THEN
         CALL errore('create_basis', 'Error creating basis', status_c)
      ENDIF
   end function create_basis

   FUNCTION get_basis_size(basis_ptr) result(size)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      INTEGER(c_int) :: size
      INTEGER(c_int), target :: size_c
      INTEGER(c_int) :: status

      status = c_spir_basis_get_size(basis_ptr, c_loc(size_c))
      IF (status /= 0) THEN
         CALL errore('get_basis_size', 'Error getting basis size', status)
      ENDIF
      size = size_c
   END FUNCTION get_basis_size

   FUNCTION basis_get_svals(basis_ptr) result(svals)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      REAL(KIND = DP), ALLOCATABLE :: svals(:)
      INTEGER(c_int), target :: nsvals_c
      INTEGER(c_int) :: status
      REAL(c_double), ALLOCATABLE, target :: svals_c(:)

      status = c_spir_basis_get_size(basis_ptr, c_loc(nsvals_c))
      IF (status /= 0) THEN
         CALL errore('basis_get_svals', 'Error getting number of singular values', status)
      ENDIF

      ALLOCATE(svals_c(nsvals_c))

      status = c_spir_basis_get_svals(basis_ptr, c_loc(svals_c))
      IF (status /= 0) THEN
         CALL errore('basis_get_svals', 'Error getting singular values', status)
      ENDIF

      ALLOCATE(svals(nsvals_c))
      svals = svals_c

      DEALLOCATE(svals_c)
   END FUNCTION basis_get_svals

   SUBROUTINE create_tau_smpl(basis_ptr, tau, tau_smpl_ptr)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      REAL(KIND = DP), ALLOCATABLE, INTENT(OUT) :: tau(:)
      TYPE(c_ptr), INTENT(OUT) :: tau_smpl_ptr

      INTEGER(c_int), target :: ntau_c
      INTEGER(c_int), target :: status_c
      REAL(c_double), ALLOCATABLE, target :: tau_c(:)

      status_c = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau_c))
      IF (status_c /= 0) THEN
         CALL errore('create_tau_smpl', 'Error getting number of tau points', status_c)
      ENDIF

      ALLOCATE(tau_c(ntau_c))
      if (allocated(tau)) deallocate(tau)
      ALLOCATE(tau(ntau_c))

      status_c = c_spir_basis_get_default_taus(basis_ptr, c_loc(tau_c))
      IF (status_c /= 0) THEN
         CALL errore('create_tau_smpl', 'Error getting tau points', status_c)
      ENDIF
      tau = REAL(tau_c, KIND=8)

      tau_smpl_ptr = c_spir_tau_sampling_new(basis_ptr, ntau_c, c_loc(tau_c), c_loc(status_c))
      IF (status_c /= 0) THEN
         CALL errore('create_tau_smpl', 'Error creating tau sampling points', status_c)
      ENDIF

      DEALLOCATE(tau_c)
   END SUBROUTINE create_tau_smpl

   SUBROUTINE create_matsu_smpl(basis_ptr, positive_only, matsus, matsu_smpl_ptr)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      LOGICAL, INTENT(IN) :: positive_only
      INTEGER(8), ALLOCATABLE, INTENT(OUT) :: matsus(:)
      TYPE(c_ptr), INTENT(OUT) :: matsu_smpl_ptr

      INTEGER(c_int), target :: nfreq_c
      INTEGER(c_int), target :: status_c
      INTEGER(c_int64_t), ALLOCATABLE, target :: matsus_c(:)
      INTEGER(c_int) :: positive_only_c

      positive_only_c = MERGE(1, 0, positive_only)

      status_c = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only_c, c_loc(nfreq_c))
      IF (status_c /= 0) THEN
         CALL errore('create_matsu_smpl', 'Error getting number of fermionic frequencies', status_c)
      ENDIF

      ALLOCATE(matsus_c(nfreq_c))

      status_c = c_spir_basis_get_default_matsus(basis_ptr, positive_only_c, c_loc(matsus_c))
      IF (status_c /= 0) THEN
         CALL errore('create_matsu_smpl', 'Error getting frequencies', status_c)
      ENDIF

      if (ALLOCATED(matsus)) DEALLOCATE(matsus)
      ALLOCATE(matsus(nfreq_c))
      matsus = matsus_c

      DEALLOCATE(matsus_c)
   END SUBROUTINE create_matsu_smpl


   FUNCTION basis_get_ws(basis_ptr) result(ws)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      REAL(KIND = DP), ALLOCATABLE :: ws(:)
      INTEGER(c_int), target :: nomega_c
      INTEGER(c_int) :: status
      REAL(c_double), ALLOCATABLE, target :: ws_c(:)

      status = c_spir_basis_get_n_default_ws(basis_ptr, c_loc(nomega_c))
      IF (status /= 0) THEN
         CALL errore('basis_get_ws', 'Error getting number of real frequencies', status)
      ENDIF

      ALLOCATE(ws_c(nomega_c))

      status = c_spir_basis_get_default_ws(basis_ptr, c_loc(ws_c))
      IF (status /= 0) THEN
         CALL errore('basis_get_ws', 'Error getting real frequencies', status)
      ENDIF

      ALLOCATE(ws(nomega_c))
      ws = ws_c

      DEALLOCATE(ws_c)
   END FUNCTION basis_get_ws

   !-----------------------------------------------------------------------
   SUBROUTINE init_ir(obj, beta, lambda, eps, positive_only)
      !-----------------------------------------------------------------------
      !!
      !! This routine initializes arrays related to the IR-basis objects.
      !! This routine should be called by read_ir or mk_ir_preset.
      !! Do not call in other routines directly.
      !!
      !
      TYPE(IR), INTENT(INOUT) :: obj
      !! contains all the IR-basis objects
      REAL(KIND = DP), INTENT(IN) :: beta
      !! inverse temperature
      REAL(KIND = DP), INTENT(IN) :: lambda
      !! lambda = 10^{nlambda}
      REAL(KIND = DP), INTENT(IN) :: eps
      !! cutoff for the singular value expansion
      LOGICAL, INTENT(IN) :: positive_only
      !! if true, take the Matsubara frequencies
      !! only from the positive region

      REAL(KIND = DP) :: wmax

      TYPE(c_ptr) :: sve_ptr, basis_f_ptr, basis_b_ptr, k_ptr

      wmax = lambda / beta

      k_ptr = create_logistic_kernel(lambda)
      if (.not. C_ASSOCIATED(k_ptr)) then
         CALL errore('init_ir', 'Kernel is not assigned', 1)
      end if

      sve_ptr = create_sve_result(lambda, eps, k_ptr)
      if (.not. C_ASSOCIATED(sve_ptr)) then
         CALL errore('init_ir', 'SVE result is not assigned', 1)
      end if

      basis_f_ptr = create_basis(SPIR_STATISTICS_FERMIONIC, beta, wmax, k_ptr, sve_ptr)
      if (.not. C_ASSOCIATED(basis_f_ptr)) then
         CALL errore('init_ir', 'Fermionic basis is not assigned', 1)
      end if

      basis_b_ptr = create_basis(SPIR_STATISTICS_BOSONIC, beta, wmax, k_ptr, sve_ptr)
      if (.not. C_ASSOCIATED(basis_b_ptr)) then
         CALL errore('init_ir', 'Bosonic basis is not assigned', 1)
      end if

      obj%basis_f_ptr = basis_f_ptr
      obj%basis_b_ptr = basis_b_ptr
      obj%sve_ptr = sve_ptr
      obj%k_ptr = k_ptr

      obj%size = get_basis_size(basis_f_ptr)

      call create_tau_smpl(basis_f_ptr, obj%tau, obj%tau_smpl_ptr)

      obj%s = basis_get_svals(basis_f_ptr)
      obj%ntau = size(obj%tau)
      call create_matsu_smpl(basis_f_ptr, positive_only, obj%freq_f, obj%matsu_f_smpl_ptr)
      call create_matsu_smpl(basis_b_ptr, positive_only, obj%freq_b, obj%matsu_b_smpl_ptr)
      obj%nfreq_f = size(obj%freq_f)
      obj%nfreq_b = size(obj%freq_b)
      obj%omega = basis_get_ws(basis_f_ptr)
      obj%nomega = size(obj%omega)

   END SUBROUTINE


   SUBROUTINE finalize_ir(obj)
      !-----------------------------------------------------------------------
      !!
      !! This routine deallocates IR-basis objects contained in obj
      !!
      !
      TYPE(IR) :: obj
      !! contains all the IR-basis objects
      INTEGER :: ierr
      !! Error status
      !
      ! Deallocate all member variables
      DEALLOCATE(obj%s, STAT = ierr)
      IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%s', 1)
      DEALLOCATE(obj%tau, STAT = ierr)
      IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%tau', 1)
      DEALLOCATE(obj%omega, STAT = ierr)
      IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%omega', 1)
      DEALLOCATE(obj%freq_f, STAT = ierr)
      IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%freq_f', 1)
      DEALLOCATE(obj%freq_b, STAT = ierr)
      IF (ierr /= 0) CALL errore('finalize_ir', 'Error deallocating IR%freq_b', 1)
      !-----------------------------------------------------------------------

      if (C_ASSOCIATED(obj%basis_f_ptr)) then
         CALL c_spir_basis_release(obj%basis_f_ptr)
      end if
      if (C_ASSOCIATED(obj%basis_b_ptr)) then
         CALL c_spir_basis_release(obj%basis_b_ptr)
      end if
      if (C_ASSOCIATED(obj%sve_ptr)) then
         CALL c_spir_sve_result_release(obj%sve_ptr)
      end if
      if (C_ASSOCIATED(obj%k_ptr)) then
         CALL c_spir_kernel_release(obj%k_ptr)
      end if
      if (C_ASSOCIATED(obj%tau_smpl_ptr)) then
         CALL c_spir_sampling_release(obj%tau_smpl_ptr)
      end if
      if (C_ASSOCIATED(obj%matsu_f_smpl_ptr)) then
         CALL c_spir_sampling_release(obj%matsu_f_smpl_ptr)
      end if
      if (C_ASSOCIATED(obj%matsu_b_smpl_ptr)) then
         CALL c_spir_sampling_release(obj%matsu_b_smpl_ptr)
      end if
   END SUBROUTINE finalize_ir

   SUBROUTINE errore(routine, msg, ierr)
      !-----------------------------------------------------------------------
      !!
      !! This routine handles error messages and program termination
      !!
      !
      IMPLICIT NONE
      CHARACTER(*), INTENT(IN) :: routine, msg
      INTEGER, INTENT(IN) :: ierr
      !
      ! Print error message with asterisk border
      WRITE(UNIT = 0, FMT = '(/,1X,78("*"))')
      WRITE(UNIT = 0, FMT = '(5X,"from ",A," : error #",I10)') routine, ierr
      WRITE(UNIT = 0, FMT = '(5X,A)') msg
      WRITE(UNIT = 0, FMT = '(1X,78("*"),/)')
      !
      STOP
      !-----------------------------------------------------------------------
   END SUBROUTINE errore


   subroutine flatten_dd(x, flat)
      REAL(KIND = DP), INTENT(IN) :: x(..) ! Arbitrary rank
      REAL(KIND = DP), ALLOCATABLE, INTENT(OUT) :: flat(:) ! 1D array

      if (allocated(flat)) deallocate(flat)

      select rank(x)
       rank(1)
         flat = x
       rank(2)
         flat = reshape(x, [size(x)])
       rank(3)
         flat = reshape(x, [size(x)])
       rank(4)
         flat = reshape(x, [size(x)])
       rank(5)
         flat = reshape(x, [size(x)])
       rank(6)
         flat = reshape(x, [size(x)])
       rank(7)
       rank default
         print *, "Error: Unsupported rank", rank(x)
         stop
      end select
   end subroutine

   subroutine flatten_zz(x, flat)
      COMPLEX(KIND = DP), INTENT(IN) :: x(..) ! Arbitrary rank
      COMPLEX(KIND = DP), ALLOCATABLE, INTENT(OUT) :: flat(:) ! 1D array

      if (allocated(flat)) deallocate(flat)

      select rank(x)
       rank(1)
         flat = x
       rank(2)
         flat = reshape(x, [size(x)])
       rank(3)
         flat = reshape(x, [size(x)])
       rank(4)
         flat = reshape(x, [size(x)])
       rank(5)
         flat = reshape(x, [size(x)])
       rank(6)
         flat = reshape(x, [size(x)])
       rank(7)
         flat = reshape(x, [size(x)])
       rank default
         print *, "Error: Unsupported rank", rank(x)
         stop
      end select
   end subroutine

   subroutine flatten_zd(x, flat)
      COMPLEX(KIND = DP), INTENT(IN) :: x(..) ! Arbitrary rank
      REAL(KIND = DP), ALLOCATABLE, INTENT(OUT) :: flat(:) ! 1D array

      if (allocated(flat)) deallocate(flat)

      select rank(x)
       rank(1)
         flat = real(x, kind=DP)
       rank(2)
         flat = reshape(real(x, kind=DP), [size(x)])
       rank(3)
         flat = reshape(real(x, kind=DP), [size(x)])
       rank(4)
         flat = reshape(real(x, kind=DP), [size(x)])
       rank(5)
         flat = reshape(real(x, kind=DP), [size(x)])
       rank(6)
         flat = reshape(real(x, kind=DP), [size(x)])
       rank(7)
         flat = reshape(real(x, kind=DP), [size(x)])
       rank default
         print *, "Error: Unsupported rank", rank(x)
         stop
      end select
   end subroutine

   subroutine unflatten_zz(flat, x)
      COMPLEX(KIND = DP), INTENT(IN) :: flat(:) ! 1D array
      COMPLEX(KIND = DP), INTENT(OUT) :: x(..) ! Arbitrary rank

      select rank(x)
       rank(1)
         x = flat
       rank(2)
         x = reshape(flat, shape(x))
       rank(3)
         x = reshape(flat, shape(x))
       rank(4)
         x = reshape(flat, shape(x))
       rank(5)
         x = reshape(flat, shape(x))
       rank(6)
         x = reshape(flat, shape(x))
       rank(7)
         x = reshape(flat, shape(x))
       rank default
         print *, "Error: Unsupported rank", rank(x)
         stop
      end select
   end subroutine unflatten_zz

   subroutine unflatten_dd(flat, x)
      REAL(KIND = DP), INTENT(IN) :: flat(:) ! 1D array
      REAL(KIND = DP), INTENT(OUT) :: x(..) ! Arbitrary rank

      select rank(x)
       rank(1)
         x = flat
       rank(2)
         x = reshape(flat, shape(x))
       rank(3)
         x = reshape(flat, shape(x))
       rank(4)
         x = reshape(flat, shape(x))
       rank(5)
         x = reshape(flat, shape(x))
       rank(6)
         x = reshape(flat, shape(x))
       rank(7)
         x = reshape(flat, shape(x))
       rank default
         print *, "Error: Unsupported rank", rank(x)
         stop
      end select
   end subroutine unflatten_dd

   subroutine unflatten_dz(flat, x)
      REAL(KIND = DP), INTENT(IN) :: flat(:) ! 1D array
      COMPLEX(KIND = DP), INTENT(OUT) :: x(..) ! Arbitrary rank

      select rank(x)
       rank(1)
         x = cmplx(flat, 0.0_DP, kind=DP)
       rank(2)
         x = reshape(cmplx(flat, 0.0_DP, kind=DP), shape(x))
       rank(3)
         x = reshape(cmplx(flat, 0.0_DP, kind=DP), shape(x))
       rank(4)
         x = reshape(cmplx(flat, 0.0_DP, kind=DP), shape(x))
       rank(5)
         x = reshape(cmplx(flat, 0.0_DP, kind=DP), shape(x))
       rank(6)
         x = reshape(cmplx(flat, 0.0_DP, kind=DP), shape(x))
       rank(7)
         x = reshape(cmplx(flat, 0.0_DP, kind=DP), shape(x))
       rank default
         print *, "Error: Unsupported rank", rank(x)
         stop
      end select
   end subroutine unflatten_dz

   function check_output_dims(target_dim, input_dims, output_dims) result(is_valid)
      INTEGER, INTENT(IN) :: target_dim
      INTEGER(c_int), INTENT(IN) :: input_dims(:)
      INTEGER(c_int), INTENT(IN) :: output_dims(:)
      LOGICAL :: is_valid

      INTEGER :: i

      do i = 1, size(input_dims)
         if (i == target_dim) then
            continue
         end if
         if (input_dims(i) /= output_dims(i)) then
            is_valid = .false.
            return
         end if
      end do
      is_valid = .true.
   end function check_output_dims

   SUBROUTINE evaluate_tau_zz(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, target :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('evaluate_tau_zz', 'Target dimension is out of range', 1)
      end if

      if (input_dims_c(target_dim) /= obj%size) then
         CALL errore('evaluate_tau_zz', 'Target dimension is not the same as the basis size', 1)
      end if

      if (output_dims_c(target_dim) /= size(obj%tau)) then
         CALL errore('evaluate_tau_zz', 'Target dimension is not the same as the number of tau sampling points', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('evaluate_tau_zz', 'Output dimensions are not the same as the input dimensions except for the target dimension', 1)
      end if

      target_dim_c = target_dim - 1
      IF (obj%positive_only) THEN
         BLOCK
            REAL(c_double), allocatable, target :: arr_c(:), res_c(:)
            call flatten_zd(arr, arr_c)

            status_c = c_spir_sampling_eval_dd(obj%tau_smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
               ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

            if (status_c /= 0) then
               CALL errore('evaluate_tau_zz', 'Error evaluating on tau sampling points', status_c)
            end if

            call unflatten_dz(res_c, res)
         END BLOCK
      ELSE
         BLOCK
            COMPLEX(c_double), allocatable, target :: arr_c(:), res_c(:)
            call flatten_zz(arr, arr_c)

            status_c = c_spir_sampling_eval_zz(obj%tau_smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
               ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

            if (status_c /= 0) then
               CALL errore('evaluate_tau_zz', 'Error evaluating on tau sampling points', status_c)
            end if

            call unflatten_zz(res_c, res)
         END BLOCK
      END IF
   END SUBROUTINE evaluate_tau_zz


   SUBROUTINE evaluate_tau_dd(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      REAL(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, target :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      REAL(c_double), allocatable, target :: arr_c(:), res_c(:)

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in output_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('evaluate_tau_dd', 'Target dimension is out of range', 1)
      end if

      if (input_dims_c(target_dim) /= obj%size) then
         CALL errore('evaluate_tau_zz', 'Target dimension is not the same as the basis size', 1)
      end if

      if (output_dims_c(target_dim) /= size(obj%tau)) then
         CALL errore('evaluate_tau_zz', 'Target dimension is not the same as the number of tau sampling points', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('evaluate_tau_zz', 'Output dimensions are not the same as the input dimensions except for the target dimension', 1)
      end if

      call flatten_dd(arr, arr_c)

      target_dim_c = target_dim - 1
      status_c = c_spir_sampling_eval_dd(obj%tau_smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
         ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

      if (status_c /= 0) then
         CALL errore('evaluate_tau_dd', 'Error evaluating on tau sampling points', status_c)
      end if

      call unflatten_dd(res_c, res)
      deallocate(arr_c, res_c)
   END SUBROUTINE evaluate_tau_dd

   SUBROUTINE evaluate_matsubara_zz(smpl_ptr, target_dim, arr, res)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, target :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('evaluate_matsubara_zz', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('evaluate_matsubara_zz', 'Output dimensions are not the same as the input dimensions except for the target dimension', 1)
      end if

      target_dim_c = target_dim - 1
      BLOCK
         COMPLEX(c_double), allocatable, target :: arr_c(:), res_c(:)
         call flatten_zz(arr, arr_c)

         status_c = c_spir_sampling_eval_zz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            CALL errore('evaluate_matsubara_zz', 'Error evaluating on Matsubara frequencies', status_c)
         end if

         call unflatten_zz(res_c, res)
      END BLOCK
   END SUBROUTINE evaluate_matsubara_zz

   SUBROUTINE evaluate_matsubara_dz(smpl_ptr, target_dim, arr, res)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, target :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      REAL(c_double), allocatable, target :: arr_c(:), res_c(:)

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in output_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('evaluate_matsubara_dz', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('evaluate_matsubara_dz', 'Output dimensions are not the same as the input dimensions except for the target dimension', 1)
      end if

      call flatten_dd(arr, arr_c)

      target_dim_c = target_dim - 1
      status_c = c_spir_sampling_eval_dz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
         ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

      if (status_c /= 0) then
         CALL errore('evaluate_matsubara_dz', 'Error evaluating on Matsubara frequencies', status_c)
      end if

      call unflatten_dz(res_c, res)
      deallocate(arr_c, res_c)
   END SUBROUTINE evaluate_matsubara_dz

   SUBROUTINE evaluate_matsubara_f_zz(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      call evaluate_matsubara_zz(obj%matsu_f_smpl_ptr, target_dim, arr, res)
   END SUBROUTINE evaluate_matsubara_f_zz

   SUBROUTINE evaluate_matsubara_f_dz(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      call evaluate_matsubara_dz(obj%matsu_f_smpl_ptr, target_dim, arr, res)
   END SUBROUTINE evaluate_matsubara_f_dz

   SUBROUTINE evaluate_matsubara_b_zz(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      call evaluate_matsubara_zz(obj%matsu_b_smpl_ptr, target_dim, arr, res)
   END SUBROUTINE evaluate_matsubara_b_zz

   SUBROUTINE evaluate_matsubara_b_dz(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      call evaluate_matsubara_dz(obj%matsu_b_smpl_ptr, target_dim, arr, res)
   END SUBROUTINE evaluate_matsubara_b_dz

   SUBROUTINE fit_tau_zz(smpl_ptr, target_dim, arr, res)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, target :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('fit_tau_zz', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('fit_tau_zz', 'Output dimensions are not the same as the input dimensions except for the target dimension', 1)
      end if

      target_dim_c = target_dim - 1
      BLOCK
         COMPLEX(c_double), allocatable, target :: arr_c(:), res_c(:)
         call flatten_zz(arr, arr_c)

         status_c = c_spir_sampling_fit_zz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            CALL errore('fit_tau_zz', 'Error fitting on tau sampling points', status_c)
         end if

         call unflatten_zz(res_c, res)
      END BLOCK
   END SUBROUTINE fit_tau_zz

   SUBROUTINE fit_tau_dd(smpl_ptr, target_dim, arr, res)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      REAL(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, target :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      REAL(c_double), allocatable, target :: arr_c(:), res_c(:)

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in output_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('fit_tau_dd', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('fit_tau_dd', 'Output dimensions are not the same as the input dimensions except for the target dimension', 1)
      end if

      call flatten_dd(arr, arr_c)

      target_dim_c = target_dim - 1
      status_c = c_spir_sampling_fit_dd(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
         ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

      if (status_c /= 0) then
         CALL errore('fit_tau_dd', 'Error fitting on tau sampling points', status_c)
      end if

      call unflatten_dd(res_c, res)
      deallocate(arr_c, res_c)
   END SUBROUTINE fit_tau_dd

   SUBROUTINE fit_matsubara_zz(smpl_ptr, target_dim, arr, res)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, target :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('fit_matsubara_zz', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('fit_matsubara_zz', 'Output dimensions are not the same as the input dimensions except for the target dimension', 1)
      end if

      target_dim_c = target_dim - 1
      BLOCK
         COMPLEX(c_double), allocatable, target :: arr_c(:), res_c(:)
         call flatten_zz(arr, arr_c)

         status_c = c_spir_sampling_fit_zz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            CALL errore('fit_matsubara_zz', 'Error fitting on Matsubara frequencies', status_c)
         end if

         call unflatten_zz(res_c, res)
      END BLOCK
   END SUBROUTINE fit_matsubara_zz

   SUBROUTINE fit_matsubara_f_zz(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      COMPLEX(KIND = DP), ALLOCATABLE :: temp(:)

      call fit_matsubara_zz(obj%matsu_f_smpl_ptr, target_dim, arr, res)
      if (obj%positive_only) then
         call flatten_zz(res, temp)
         temp = CMPLX(REAL(temp, KIND=DP), 0.0_DP, KIND=DP)
         call unflatten_zz(temp, res)
         deallocate(temp)
      end if
   END SUBROUTINE fit_matsubara_f_zz

   SUBROUTINE fit_matsubara_b_zz(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      COMPLEX(KIND = DP), ALLOCATABLE :: temp(:)

      call fit_matsubara_zz(obj%matsu_b_smpl_ptr, target_dim, arr, res)
      if (obj%positive_only) then
         call flatten_zz(res, temp)
         temp = CMPLX(REAL(temp, KIND=DP), 0.0_DP, KIND=DP)
         call unflatten_zz(temp, res)
         deallocate(temp)
      end if
   END SUBROUTINE fit_matsubara_b_zz

END MODULE sparseir_ext
