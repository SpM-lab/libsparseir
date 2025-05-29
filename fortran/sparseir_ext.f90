! Extends the SparseIR library with additional functionality
MODULE sparseir_ext
   use, intrinsic :: iso_c_binding
   use sparseir
   implicit none

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
      DOUBLE PRECISION :: beta
      !! inverse temperature
      DOUBLE PRECISION :: lambda
      !! lambda = 10^{nlambda},
      !! which determines maximum sampling point of real frequency
      DOUBLE PRECISION :: wmax
      !! maximum real frequency: wmax = lambda / beta
      DOUBLE PRECISION :: eps
      !! eps = 10^{-ndigit}
      DOUBLE PRECISION :: eps_svd
      !! This is used in the SVD fitting.
      DOUBLE PRECISION, ALLOCATABLE :: s(:)
      !! singular values
      DOUBLE PRECISION, ALLOCATABLE :: tau(:)
      !! sampling points of imaginary time
      DOUBLE PRECISION, ALLOCATABLE :: x(:)
      !! This is used to get tau: tau = 5.0d-1 * beta * (x + 1.d0)
      DOUBLE PRECISION, ALLOCATABLE :: omega(:)
      !! sampling points of real frequency
      DOUBLE PRECISION, ALLOCATABLE :: y(:)
      !! This is used to get omega: omega = y * wmax
      INTEGER(8), ALLOCATABLE :: freq_f(:)
      !! integer part of sampling Matsubara freqs (Fermion)
      INTEGER(8), ALLOCATABLE :: freq_b(:)
      !! integer part of sampling Matsubara freqs (Boson)
      LOGICAL :: positive_only
      !! if true, take the Matsubara frequencies
      !! only from the positive region

      type(c_ptr) :: basis_f_ptr
      !! pointer to the fermionic basis
      type(c_ptr) :: basis_b_ptr
      !! pointer to the bosonic basis
      type(c_ptr) :: sve_ptr
      !! pointer to the SVE result
      type(c_ptr) :: k_ptr
      !! pointer to the kernel
      !-----------------------------------------------------------------------
   END TYPE IR
   !-----------------------------------------------------------------------

contains
   FUNCTION create_logistic_kernel(lambda) result(k_ptr)
      DOUBLE PRECISION, INTENT(IN) :: lambda
      REAL(c_double), target :: lambda_c
      INTEGER(c_int), target :: status_c
      type(c_ptr) :: k_ptr
      lambda_c = lambda
      k_ptr = c_spir_logistic_kernel_new(lambda_c, c_loc(status_c))
   end function create_logistic_kernel

   FUNCTION create_sve_result(lambda, eps, k_ptr) result(sve_ptr)
      DOUBLE PRECISION, INTENT(IN) :: lambda
      DOUBLE PRECISION, INTENT(IN) :: eps

      REAL(c_double), target :: lambda_c, eps_c
      INTEGER(c_int), target :: status_c

      type(c_ptr), INTENT(IN) :: k_ptr
      type(c_ptr) :: sve_ptr

      lambda_c = lambda
      eps_c = eps

      sve_ptr = c_spir_sve_result_new(k_ptr, eps_c, c_loc(status_c))
      IF (status_c /= 0) THEN
         PRINT*, "Error creating SVE result"
         STOP
      ENDIF
   end function create_sve_result

   FUNCTION create_basis(statistics, beta, wmax, k_ptr, sve_ptr) result(basis_ptr)
      INTEGER, INTENT(IN) :: statistics
      DOUBLE PRECISION, INTENT(IN) :: beta
      DOUBLE PRECISION, INTENT(IN) :: wmax
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
         PRINT*, "Error creating basis"
         STOP
      ENDIF
   end function create_basis


   FUNCTION get_basis_size(basis_ptr) result(size)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      INTEGER(c_int) :: size
      INTEGER(c_int), target :: size_c
      INTEGER(c_int) :: status

      status = c_spir_basis_get_size(basis_ptr, c_loc(size_c))
      IF (status /= 0) THEN
         PRINT*, "Error getting basis size"
         STOP
      ENDIF
      size = size_c
   END FUNCTION get_basis_size

   FUNCTION basis_get_taus(basis_ptr) result(tau)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      DOUBLE PRECISION, ALLOCATABLE :: tau(:)
      INTEGER(c_int), target :: ntau_c
      INTEGER(c_int) :: status
      REAL(c_double), ALLOCATABLE, target :: tau_c(:)  ! 動的に割り当てる配列

      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau_c))
      IF (status /= 0) THEN
         PRINT*, "Error getting number of tau points"
         STOP
      ENDIF

      ALLOCATE(tau_c(ntau_c))

      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(tau_c))
      IF (status /= 0) THEN
         PRINT*, "Error getting tau points"
         STOP
      ENDIF

      ALLOCATE(tau(ntau_c))
      tau = REAL(tau_c, KIND=8)

      DEALLOCATE(tau_c)
   END FUNCTION basis_get_taus

   FUNCTION basis_get_matsus(basis_ptr, positive_only) result(matsus)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      LOGICAL, INTENT(IN) :: positive_only
      INTEGER(8), ALLOCATABLE :: matsus(:)
      INTEGER(c_int), target :: nfreq_f_c
      INTEGER(c_int) :: status
      INTEGER(c_int64_t), ALLOCATABLE, target :: matsus_c(:)
      INTEGER(c_int) :: positive_only_c

      positive_only_c = MERGE(1, 0, positive_only)

      status = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only_c, c_loc(nfreq_f_c))
      IF (status /= 0) THEN
         PRINT*, "Error getting number of fermionic frequencies"
         STOP
      ENDIF

      ALLOCATE(matsus_c(nfreq_f_c))

      status = c_spir_basis_get_default_matsus(basis_ptr, positive_only_c, c_loc(matsus_c))
      IF (status /= 0) THEN
         PRINT*, "Error getting fermionic frequencies"
         STOP
      ENDIF

      ALLOCATE(matsus(nfreq_f_c))
      matsus = matsus_c

      DEALLOCATE(matsus_c)
   END FUNCTION basis_get_matsus

   FUNCTION basis_get_ws(basis_ptr) result(ws)
      TYPE(c_ptr), INTENT(IN) :: basis_ptr
      DOUBLE PRECISION, ALLOCATABLE :: ws(:)
      INTEGER(c_int), target :: nomega_c
      INTEGER(c_int) :: status
      REAL(c_double), ALLOCATABLE, target :: ws_c(:)

      status = c_spir_basis_get_n_default_ws(basis_ptr, c_loc(nomega_c))
      IF (status /= 0) THEN
         PRINT*, "Error getting number of real frequencies"
         STOP
      ENDIF

      ALLOCATE(ws_c(nomega_c))

      status = c_spir_basis_get_default_ws(basis_ptr, c_loc(ws_c))
      IF (status /= 0) THEN
         PRINT*, "Error getting real frequencies"
         STOP
      ENDIF

      ALLOCATE(ws(nomega_c))
      ws = ws_c

      DEALLOCATE(ws_c)
   END FUNCTION basis_get_ws

   !SUBROUTINE mod_create_sve_result(lambda, eps)
   !DOUBLE PRECISION, INTENT(IN) :: lambda
   !DOUBLE PRECISION, INTENT(IN) :: eps
!
   !sve_ptr = create_sve_result(lambda, eps)
   !END SUBROUTINE mod_create_sve_result

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
      DOUBLE PRECISION, INTENT(IN) :: beta
      !! inverse temperature
      DOUBLE PRECISION, INTENT(IN) :: lambda
      !! lambda = 10^{nlambda}
      DOUBLE PRECISION, INTENT(IN) :: eps
      !! cutoff for the singular value expansion
      LOGICAL, INTENT(IN) :: positive_only
      !! if true, take the Matsubara frequencies
      !! only from the positive region

      DOUBLE PRECISION :: wmax

      type(c_ptr) :: sve_ptr, basis_f_ptr, basis_b_ptr, k_ptr

      wmax = lambda / beta

      k_ptr = create_logistic_kernel(lambda)
      if (.not. c_associated(k_ptr)) then
         print*, "Error: Kernel is not assigned"
         stop
      else
         print*, "Kernel is assigned"
      end if

      sve_ptr = create_sve_result(lambda, eps, k_ptr)
      if (.not. c_associated(sve_ptr)) then
         print*, "Error: SVE result is not assigned"
         stop
      else
         print*, "SVE result is assigned"
      end if

      basis_f_ptr = create_basis(SPIR_STATISTICS_FERMIONIC, beta, wmax, k_ptr, sve_ptr)
      if (.not. c_associated(basis_f_ptr)) then
         print*, "Error: Basis is not assigned"
         stop
      else
         print*, "Basis is assigned"
      end if

      basis_b_ptr = create_basis(SPIR_STATISTICS_BOSONIC, beta, wmax, k_ptr, sve_ptr)
      if (.not. c_associated(basis_b_ptr)) then
         print*, "Error: Basis is not assigned"
         stop
      else
         print*, "Basis is assigned"
      end if

      obj%basis_f_ptr = basis_f_ptr
      obj%basis_b_ptr = basis_b_ptr
      obj%sve_ptr = sve_ptr
      obj%k_ptr = k_ptr

      obj%size = get_basis_size(basis_f_ptr)
      obj%tau = basis_get_taus(basis_f_ptr)
      obj%ntau = size(obj%tau)
      obj%freq_f = basis_get_matsus(basis_f_ptr, positive_only)
      obj%freq_b = basis_get_matsus(basis_b_ptr, positive_only)
      obj%nfreq_f = size(obj%freq_f)
      obj%nfreq_b = size(obj%freq_b)
      obj%omega = basis_get_ws(basis_f_ptr)
      obj%nomega = size(obj%omega)

   END SUBROUTINE
END MODULE sparseir_ext
