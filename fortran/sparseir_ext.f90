! ExtENDs the SparseIR library with additional functionality
MODULE sparseir_ext
   USE, INTRINSIC :: iso_c_binding
   USE sparseir
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: IR, evaluate_tau, evaluate_matsubara, fit_tau, fit_matsubara, ir2dlr, dlr2ir
   PUBLIC :: init_ir, finalize_ir

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
      INTEGER :: npoles
      !! total number of DLR poles
      REAL(KIND = DP) :: beta
      !! inverse temperature
      REAL(KIND = DP) :: lambda
      !! lambda = 10^{nlambda},
      !! which determines maximum sampling point of real frequency
      REAL(KIND = DP) :: wmax
      !! maximum real frequency: wmax = lambda / beta
      REAL(KIND = DP) :: eps
      !! eps = 10^{-ndigit}
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
      !! IF true, take the Matsubara frequencies
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
      TYPE(c_ptr) :: dlr_f_ptr
      !! pointer to the fermionic DLR
      TYPE(c_ptr) :: dlr_b_ptr
      !! pointer to the bosonic DLR
      !-----------------------------------------------------------------------
   END TYPE IR
   !-----------------------------------------------------------------------

   INTERFACE evaluate_tau
      MODULE PROCEDURE evaluate_tau_zz_impl, evaluate_tau_dd_impl
   END INTERFACE evaluate_tau

   INTERFACE evaluate_matsubara
      MODULE PROCEDURE evaluate_matsubara_zz_impl, evaluate_matsubara_dz_impl
   END INTERFACE evaluate_matsubara

   INTERFACE fit_tau
      MODULE PROCEDURE fit_tau_zz_impl, fit_tau_dd_impl
   END INTERFACE fit_tau

   INTERFACE fit_matsubara
      MODULE PROCEDURE fit_matsubara_zz_impl
   END INTERFACE fit_matsubara

   INTERFACE ir2dlr
      MODULE PROCEDURE ir2dlr_zz_impl, ir2dlr_dd_impl
   END INTERFACE ir2dlr

   INTERFACE dlr2ir
      MODULE PROCEDURE dlr2ir_zz_impl, dlr2ir_dd_impl
   END INTERFACE dlr2ir

contains
   FUNCTION create_logistic_kernel(lambda) result(k_ptr)
      REAL(KIND = DP), INTENT(IN) :: lambda
      REAL(c_double), TARGET :: lambda_c
      INTEGER(c_int), TARGET :: status_c
      TYPE(c_ptr) :: k_ptr
      lambda_c = lambda
      k_ptr = c_spir_logistic_kernel_new(lambda_c, c_loc(status_c))
   END function create_logistic_kernel

   FUNCTION create_sve_result(lambda, eps, k_ptr) result(sve_ptr)
      REAL(KIND = DP), INTENT(IN) :: lambda
      REAL(KIND = DP), INTENT(IN) :: eps

      REAL(c_double), TARGET :: lambda_c, eps_c
      INTEGER(c_int), TARGET :: status_c

      TYPE(c_ptr), INTENT(IN) :: k_ptr
      TYPE(c_ptr) :: sve_ptr

      lambda_c = lambda
      eps_c = eps

      sve_ptr = c_spir_sve_result_new(k_ptr, eps_c, c_loc(status_c))
      IF (status_c /= 0) THEN
         CALL errore('create_sve_result', 'Error creating SVE result', status_c)
      ENDIF
   END function create_sve_result

   FUNCTION create_basis(statistics, beta, wmax, k_ptr, sve_ptr) result(basis_ptr)
      INTEGER, INTENT(IN) :: statistics
      REAL(KIND = DP), INTENT(IN) :: beta
      REAL(KIND = DP), INTENT(IN) :: wmax
      TYPE(c_ptr), INTENT(IN) :: k_ptr, sve_ptr

      INTEGER(c_int), TARGET :: status_c
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
   END function create_basis

   FUNCTION get_basis_size(basis_ptr) result(size)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      INTEGER(c_int) :: size
      INTEGER(c_int), TARGET :: size_c
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
      INTEGER(c_int), TARGET :: nsvals_c
      INTEGER(c_int) :: status
      REAL(c_double), ALLOCATABLE, TARGET :: svals_c(:)

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

      INTEGER(c_int), TARGET :: ntau_c
      INTEGER(c_int), TARGET :: status_c
      REAL(c_double), ALLOCATABLE, TARGET :: tau_c(:)

      status_c = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau_c))
      IF (status_c /= 0) THEN
         CALL errore('create_tau_smpl', 'Error getting number of tau points', status_c)
      ENDIF

      ALLOCATE(tau_c(ntau_c))
      IF (ALLOCATEd(tau)) DEALLOCATE(tau)
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

      INTEGER(c_int), TARGET :: nfreq_c
      INTEGER(c_int), TARGET :: status_c
      INTEGER(c_int64_t), ALLOCATABLE, TARGET :: matsus_c(:)
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

      IF (ALLOCATED(matsus)) DEALLOCATE(matsus)
      ALLOCATE(matsus(nfreq_c))
      matsus = matsus_c

      ! Create sampling object
      matsu_smpl_ptr = c_spir_matsu_sampling_new(basis_ptr, positive_only_c, nfreq_c, c_loc(matsus_c), c_loc(status_c))
      IF (status_c /= 0) THEN
         CALL errore('create_matsu_smpl', 'Error creating sampling object', status_c)
      ENDIF

      DEALLOCATE(matsus_c)
   END SUBROUTINE create_matsu_smpl

   FUNCTION basis_get_ws(basis_ptr) result(ws)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      REAL(KIND = DP), ALLOCATABLE :: ws(:)
      INTEGER(c_int), TARGET :: nomega_c
      INTEGER(c_int) :: status
      REAL(c_double), ALLOCATABLE, TARGET :: ws_c(:)

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


   SUBROUTINE init_ir(obj, beta, lambda, eps, positive_only)
      !-----------------------------------------------------------------------
      !!
      !! This routine initializes arrays related to the IR-basis objects.
      !! This routine should be CALLed by read_ir or mk_ir_preset.
      !! Do not CALL in other routines directly.
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
      !! IF true, take the Matsubara frequencies
      !! only from the positive region

      REAL(KIND = DP) :: wmax
      INTEGER(c_int), TARGET :: status_c, npoles_c

      TYPE(c_ptr) :: sve_ptr, basis_f_ptr, basis_b_ptr, k_ptr, dlr_f_ptr, dlr_b_ptr

      wmax = lambda / beta

      k_ptr = create_logistic_kernel(lambda)
      IF (.not. C_ASSOCIATED(k_ptr)) then
         CALL errore('init_ir', 'Kernel is not assigned', 1)
      END IF

      sve_ptr = create_sve_result(lambda, eps, k_ptr)
      IF (.not. C_ASSOCIATED(sve_ptr)) then
         CALL errore('init_ir', 'SVE result is not assigned', 1)
      END IF

      basis_f_ptr = create_basis(SPIR_STATISTICS_FERMIONIC, beta, wmax, k_ptr, sve_ptr)
      IF (.not. C_ASSOCIATED(basis_f_ptr)) then
         CALL errore('init_ir', 'Fermionic basis is not assigned', 1)
      END IF

      basis_b_ptr = create_basis(SPIR_STATISTICS_BOSONIC, beta, wmax, k_ptr, sve_ptr)
      IF (.not. C_ASSOCIATED(basis_b_ptr)) then
         CALL errore('init_ir', 'Bosonic basis is not assigned', 1)
      END IF

      ! Create DLR objects
      dlr_f_ptr = c_spir_dlr_new(basis_f_ptr, c_loc(status_c))
      IF (status_c /= 0 .or. .not. C_ASSOCIATED(dlr_f_ptr)) then
         CALL errore('init_ir', 'Error creating fermionic DLR', status_c)
      END IF

      dlr_b_ptr = c_spir_dlr_new(basis_b_ptr, c_loc(status_c))
      IF (status_c /= 0 .or. .not. C_ASSOCIATED(dlr_b_ptr)) then
         CALL errore('init_ir', 'Error creating bosonic DLR', status_c)
      END IF

      ! Get number of poles
      status_c = c_spir_dlr_get_npoles(dlr_f_ptr, c_loc(npoles_c))
      IF (status_c /= 0) then
         CALL errore('init_ir', 'Error getting number of poles', status_c)
      END IF

      obj%basis_f_ptr = basis_f_ptr
      obj%basis_b_ptr = basis_b_ptr
      obj%sve_ptr = sve_ptr
      obj%k_ptr = k_ptr
      obj%dlr_f_ptr = dlr_f_ptr
      obj%dlr_b_ptr = dlr_b_ptr
      obj%npoles = npoles_c

      obj%size = get_basis_size(basis_f_ptr)

      CALL create_tau_smpl(basis_f_ptr, obj%tau, obj%tau_smpl_ptr)

      obj%s = basis_get_svals(basis_f_ptr)
      obj%ntau = size(obj%tau)
      CALL create_matsu_smpl(basis_f_ptr, positive_only, obj%freq_f, obj%matsu_f_smpl_ptr)
      CALL create_matsu_smpl(basis_b_ptr, positive_only, obj%freq_b, obj%matsu_b_smpl_ptr)
      obj%nfreq_f = size(obj%freq_f)
      obj%nfreq_b = size(obj%freq_b)
      obj%omega = basis_get_ws(basis_f_ptr)
      obj%nomega = size(obj%omega)
      obj%eps = eps

   END SUBROUTINE


   SUBROUTINE finalize_ir(obj)
      !-----------------------------------------------------------------------
      !!
      !! This routine DEALLOCATEs IR-basis objects contained in obj
      !!
      !
      TYPE(IR) :: obj
      !! contains all the IR-basis objects
      INTEGER :: ierr
      !! Error status
      !
      ! DeALLOCATE all member variables
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

      IF (C_ASSOCIATED(obj%basis_f_ptr)) then
         CALL c_spir_basis_release(obj%basis_f_ptr)
      END IF
      IF (C_ASSOCIATED(obj%basis_b_ptr)) then
         CALL c_spir_basis_release(obj%basis_b_ptr)
      END IF
      IF (C_ASSOCIATED(obj%sve_ptr)) then
         CALL c_spir_sve_result_release(obj%sve_ptr)
      END IF
      IF (C_ASSOCIATED(obj%k_ptr)) then
         CALL c_spir_kernel_release(obj%k_ptr)
      END IF
      IF (C_ASSOCIATED(obj%tau_smpl_ptr)) then
         CALL c_spir_sampling_release(obj%tau_smpl_ptr)
      END IF
      IF (C_ASSOCIATED(obj%matsu_f_smpl_ptr)) then
         CALL c_spir_sampling_release(obj%matsu_f_smpl_ptr)
      END IF
      IF (C_ASSOCIATED(obj%matsu_b_smpl_ptr)) then
         CALL c_spir_sampling_release(obj%matsu_b_smpl_ptr)
      END IF
      IF (C_ASSOCIATED(obj%dlr_f_ptr)) then
         CALL c_spir_basis_release(obj%dlr_f_ptr)
      END IF
      IF (C_ASSOCIATED(obj%dlr_b_ptr)) then
         CALL c_spir_basis_release(obj%dlr_b_ptr)
      END IF
   END SUBROUTINE finalize_ir

   SUBROUTINE errore(routine, msg, ierr)
      !-----------------------------------------------------------------------
      !!
      !! This routine handles error messages and program termination
      !!
      !
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

      IF (ALLOCATEd(flat)) DEALLOCATE(flat)

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
      END select
   END subroutine

   subroutine flatten_zz(x, flat)
      COMPLEX(KIND = DP), INTENT(IN) :: x(..) ! Arbitrary rank
      COMPLEX(KIND = DP), ALLOCATABLE, INTENT(OUT) :: flat(:) ! 1D array

      IF (ALLOCATEd(flat)) DEALLOCATE(flat)

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
      END select
   END subroutine

   subroutine flatten_zd(x, flat)
      COMPLEX(KIND = DP), INTENT(IN) :: x(..) ! Arbitrary rank
      REAL(KIND = DP), ALLOCATABLE, INTENT(OUT) :: flat(:) ! 1D array

      IF (ALLOCATEd(flat)) DEALLOCATE(flat)

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
      END select
   END subroutine

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
      END select
   END subroutine unflatten_zz

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
      END select
   END subroutine unflatten_dd

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
      END select
   END subroutine unflatten_dz

   function check_output_dims(target_dim, input_dims, output_dims) result(is_valid)
      INTEGER, INTENT(IN) :: target_dim
      INTEGER(c_int), INTENT(IN) :: input_dims(:)
      INTEGER(c_int), INTENT(IN) :: output_dims(:)
      LOGICAL :: is_valid

      INTEGER :: i

      if (size(input_dims) /= size(output_dims)) then
         write(*, *) "input_dims and output_dims have different sizes"
         stop  
      end if

      do i = 1, size(input_dims)
         IF (i == target_dim) then
            cycle
         END IF
         IF (input_dims(i) /= output_dims(i)) then
            print *, "input_dims(", i, ")", input_dims(i), "output_dims(", i, ")", output_dims(i)
            is_valid = .false.
            return
         END IF
      END do
      is_valid = .true.
   END function check_output_dims

   SUBROUTINE evaluate_tau_zz_impl(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('evaluate_tau_zz_impl', 'Target dimension is out of range', 1)
      END IF

      IF (input_dims_c(target_dim) /= obj%size) then
         CALL errore('evaluate_tau_zz_impl', 'Target dimension is not the same as the basis size', 1)
      END IF

      IF (output_dims_c(target_dim) /= size(obj%tau)) then
         CALL errore('evaluate_tau_zz_impl', 'Target dimension is not the same as the number of tau sampling points', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('evaluate_tau_zz_impl', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      target_dim_c = target_dim - 1
      IF (obj%positive_only) THEN
         BLOCK
            REAL(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
            CALL flatten_zd(arr, arr_c)

            ALLOCATE(res_c(PRODUCT(output_dims_c)))

            status_c = c_spir_sampling_eval_dd(obj%tau_smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
               ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

            IF (status_c /= 0) then
               CALL errore('evaluate_tau_zz_impl', 'Error evaluating on tau sampling points', status_c)
            END IF

            CALL unflatten_dz(res_c, res)
            DEALLOCATE(arr_c, res_c)
         END BLOCK
      ELSE
         BLOCK
            COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
            CALL flatten_zz(arr, arr_c)

            ALLOCATE(res_c(PRODUCT(output_dims_c)))

            status_c = c_spir_sampling_eval_zz(obj%tau_smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
               ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

            IF (status_c /= 0) then
               CALL errore('evaluate_tau_zz_impl', 'Error evaluating on tau sampling points', status_c)
            END IF

            CALL unflatten_zz(res_c, res)
            DEALLOCATE(arr_c, res_c)
         END BLOCK
      END IF
   END SUBROUTINE evaluate_tau_zz_impl


   SUBROUTINE evaluate_tau_dd_impl(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      REAL(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      REAL(c_double), allocatable, TARGET :: arr_c(:), res_c(:)

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in output_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('evaluate_tau_dd_impl', 'Target dimension is out of range', 1)
      END IF

      IF (input_dims_c(target_dim) /= obj%size) then
         CALL errore('evaluate_tau_zz_impl', 'Target dimension is not the same as the basis size', 1)
      END IF

      IF (output_dims_c(target_dim) /= size(obj%tau)) then
         CALL errore('evaluate_tau_zz_impl', 'Target dimension is not the same as the number of tau sampling points', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('evaluate_tau_zz_impl', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      CALL flatten_dd(arr, arr_c)

      target_dim_c = target_dim - 1
      status_c = c_spir_sampling_eval_dd(obj%tau_smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
         ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

      IF (status_c /= 0) then
         CALL errore('evaluate_tau_dd_impl', 'Error evaluating on tau sampling points', status_c)
      END IF

      CALL unflatten_dd(res_c, res)
      DEALLOCATE(arr_c, res_c)
   END SUBROUTINE evaluate_tau_dd_impl

   SUBROUTINE evaluate_matsubara_zz_internal(smpl_ptr, target_dim, arr, res)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('evaluate_matsubara_zz_internal', 'Target dimension is out of range', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('evaluate_matsubara_zz_internal', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF


      target_dim_c = target_dim - 1
      BLOCK
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         CALL flatten_zz(arr, arr_c)

         ALLOCATE(res_c(PRODUCT(output_dims_c)))

         status_c = c_spir_sampling_eval_zz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         IF (status_c /= 0) then
            CALL errore('evaluate_matsubara_zz_internal', 'Error evaluating on Matsubara frequencies', status_c)
         END IF

         CALL unflatten_zz(res_c, res)
         DEALLOCATE(arr_c, res_c)
      END BLOCK
   END SUBROUTINE evaluate_matsubara_zz_internal

   SUBROUTINE evaluate_matsubara_dz_internal(smpl_ptr, target_dim, arr, res)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      REAL(c_double), allocatable, TARGET :: arr_c(:), res_c(:)

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in output_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('evaluate_matsubara_dz_internal', 'Target dimension is out of range', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('evaluate_matsubara_dz_internal', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      ALLOCATE(res_c(PRODUCT(output_dims_c)))

      CALL flatten_dd(arr, arr_c)

      target_dim_c = target_dim - 1
      status_c = c_spir_sampling_eval_dz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
         ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

      IF (status_c /= 0) then
         CALL errore('evaluate_matsubara_dz_internal', 'Error evaluating on Matsubara frequencies', status_c)
      END IF

      CALL unflatten_dz(res_c, res)
      DEALLOCATE(arr_c, res_c)
   END SUBROUTINE evaluate_matsubara_dz_internal

   SUBROUTINE evaluate_matsubara_zz_impl(obj, statistics, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: statistics
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      IF (statistics == SPIR_STATISTICS_FERMIONIC) then
         CALL evaluate_matsubara_zz_internal(obj%matsu_f_smpl_ptr, target_dim, arr, res)
      ELSE IF (statistics == SPIR_STATISTICS_BOSONIC) then
         CALL evaluate_matsubara_zz_internal(obj%matsu_b_smpl_ptr, target_dim, arr, res)
      ELSE
         CALL errore('evaluate_matsubara_zz_impl', 'Invalid statistics', 1)
      END IF
   END SUBROUTINE evaluate_matsubara_zz_impl


   SUBROUTINE evaluate_matsubara_dz_impl(obj, statistics, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: statistics
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      IF (statistics == SPIR_STATISTICS_FERMIONIC) then
         CALL evaluate_matsubara_dz_internal(obj%matsu_f_smpl_ptr, target_dim, arr, res)
      ELSE IF (statistics == SPIR_STATISTICS_BOSONIC) then
         CALL evaluate_matsubara_dz_internal(obj%matsu_b_smpl_ptr, target_dim, arr, res)
      ELSE
         CALL errore('evaluate_matsubara_dz', 'Invalid statistics', 1)
      END IF
   END SUBROUTINE evaluate_matsubara_dz_impl

   SUBROUTINE fit_tau_zz_impl(smpl_ptr, target_dim, arr, res)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('fit_tau_zz_impl', 'Target dimension is out of range', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('fit_tau_zz_impl', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      target_dim_c = target_dim - 1
      BLOCK
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         CALL flatten_zz(arr, arr_c)

         ALLOCATE(res_c(PRODUCT(output_dims_c)))

         status_c = c_spir_sampling_fit_zz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         IF (status_c /= 0) then
            CALL errore('fit_tau_zz_impl', 'Error fitting on tau sampling points', status_c)
         END IF

         CALL unflatten_zz(res_c, res)
         DEALLOCATE(arr_c, res_c)
      END BLOCK
   END SUBROUTINE fit_tau_zz_impl

   SUBROUTINE fit_tau_dd_impl(smpl_ptr, target_dim, arr, res)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      REAL(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      REAL(c_double), allocatable, TARGET :: arr_c(:), res_c(:)

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in output_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('fit_tau_dd_impl', 'Target dimension is out of range', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('fit_tau_dd_impl', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      CALL flatten_dd(arr, arr_c)

      target_dim_c = target_dim - 1
      status_c = c_spir_sampling_fit_dd(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
         ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

      IF (status_c /= 0) then
         CALL errore('fit_tau_dd_impl', 'Error fitting on tau sampling points', status_c)
      END IF

      CALL unflatten_dd(res_c, res)
      DEALLOCATE(arr_c, res_c)
   END SUBROUTINE fit_tau_dd_impl

   SUBROUTINE fit_matsubara_zz_internal(smpl_ptr, target_dim, arr, res, positive_only)
      TYPE(c_ptr), INTENT(IN) :: smpl_ptr
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)
      LOGICAL, INTENT(IN), OPTIONAL :: positive_only

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('fit_matsubara_zz_internal', 'Target dimension is out of range', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('fit_matsubara_zz_internal', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      target_dim_c = target_dim - 1
      BLOCK
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         CALL flatten_zz(arr, arr_c)

         ALLOCATE(res_c(PRODUCT(output_dims_c)))

         status_c = c_spir_sampling_fit_zz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         IF (status_c /= 0) then
            CALL errore('fit_matsubara_zz_internal', 'Error fitting on Matsubara frequencies', status_c)
         END IF

         CALL unflatten_zz(res_c, res)
         DEALLOCATE(arr_c, res_c)
      END BLOCK

   END SUBROUTINE fit_matsubara_zz_internal

   SUBROUTINE fit_matsubara_zz_impl(obj, statistics, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: statistics
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      IF (statistics == SPIR_STATISTICS_FERMIONIC) then
         CALL fit_matsubara_zz_internal(obj%matsu_f_smpl_ptr, target_dim, arr, res, obj%positive_only)
      ELSE IF (statistics == SPIR_STATISTICS_BOSONIC) then
         CALL fit_matsubara_zz_internal(obj%matsu_b_smpl_ptr, target_dim, arr, res, obj%positive_only)
      ELSE
         CALL errore('fit_matsubara_zz_impl', 'Invalid statistics', 1)
      END IF
   END SUBROUTINE fit_matsubara_zz_impl


   SUBROUTINE ir2dlr_zz_impl(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('ir2dlr_zz_impl', 'Target dimension is out of range', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('ir2dlr_zz_impl', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      target_dim_c = target_dim - 1
      BLOCK
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         CALL flatten_zz(arr, arr_c)

         status_c = c_spir_ir2dlr_zz(obj%dlr_f_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         IF (status_c /= 0) then
            CALL errore('ir2dlr_zz_impl', 'Error converting IR to DLR', status_c)
         END IF

         CALL unflatten_zz(res_c, res)
      END BLOCK
   END SUBROUTINE ir2dlr_zz_impl

   SUBROUTINE ir2dlr_dd_impl(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      REAL(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('ir2dlr_dd_impl', 'Target dimension is out of range', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('ir2dlr_dd_impl', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      target_dim_c = target_dim - 1
      BLOCK
         REAL(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         CALL flatten_dd(arr, arr_c)

         status_c = c_spir_ir2dlr_dd(obj%dlr_f_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         IF (status_c /= 0) then
            CALL errore('ir2dlr_dd_impl', 'Error converting IR to DLR', status_c)
         END IF

         CALL unflatten_dd(res_c, res)
      END BLOCK
   END SUBROUTINE ir2dlr_dd_impl

   SUBROUTINE dlr2ir_zz_impl(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      COMPLEX(KIND = DP), INTENT(IN) :: arr(..)
      COMPLEX(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('dlr2ir_zz_impl', 'Target dimension is out of range', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('dlr2ir_zz_impl', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      IF (size(arr, target_dim) /= obj%npoles) then
         CALL errore('dlr2ir_zz_impl', 'Input dimension is not the same as the number of poles', 1)
      END IF

      IF (size(res, target_dim) /= obj%size) then
         CALL errore('dlr2ir_zz_impl', 'Output dimension is not the same as the size of the IR', 1)
      END IF

      target_dim_c = target_dim - 1
      BLOCK
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         CALL flatten_zz(arr, arr_c)

         ALLOCATE(res_c(PRODUCT(output_dims_c)))

         status_c = c_spir_dlr2ir_zz(obj%dlr_f_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         IF (status_c /= 0) then
            CALL errore('dlr2ir_zz_impl', 'Error converting DLR to IR', status_c)
         END IF

         CALL unflatten_zz(res_c, res)
         DEALLOCATE(arr_c, res_c)
      END BLOCK
   END SUBROUTINE dlr2ir_zz_impl

   SUBROUTINE dlr2ir_dd_impl(obj, target_dim, arr, res)
      TYPE(IR), INTENT(IN) :: obj
      INTEGER, INTENT(IN) :: target_dim
      REAL(KIND = DP), INTENT(IN) :: arr(..)
      REAL(KIND = DP), INTENT(OUT) :: res(..)

      INTEGER(c_int) :: ndim_c, target_dim_c
      INTEGER(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      INTEGER(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      IF (target_dim <= 0 .or. target_dim > ndim_c) then
         CALL errore('dlr2ir_dd_impl', 'Target dimension is out of range', 1)
      END IF

      IF (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         CALL errore('dlr2ir_dd_impl', 'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      END IF

      IF (size(arr, target_dim) /= obj%npoles) then
         CALL errore('dlr2ir_dd_impl', 'Input dimension is not the same as the number of poles', 1)
      END IF

      IF (size(res, target_dim) /= obj%size) then
         CALL errore('dlr2ir_dd_impl', 'Output dimension is not the same as the size of the IR', 1)
      END IF

      target_dim_c = target_dim - 1
      BLOCK
         REAL(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         CALL flatten_dd(arr, arr_c)

         ALLOCATE(res_c(PRODUCT(output_dims_c)))

         status_c = c_spir_dlr2ir_dd(obj%dlr_f_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         IF (status_c /= 0) then
            CALL errore('dlr2ir_dd_impl', 'Error converting DLR to IR', status_c)
         END IF

         CALL unflatten_dd(res_c, res)

         DEALLOCATE(arr_c, res_c)
      END BLOCK
   END SUBROUTINE dlr2ir_dd_impl

END MODULE sparseir_ext
