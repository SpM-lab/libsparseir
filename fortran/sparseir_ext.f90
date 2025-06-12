! Extends the SparseIR library with additional functionality
module sparseir_ext
   use, intrinsic :: iso_c_binding
   use sparseir
   implicit none
   private

   public :: IR, evaluate_tau, evaluate_matsubara, fit_tau, fit_matsubara, ir2dlr, dlr2ir
   public :: init_ir, finalize_ir, eval_u_tau

   integer, parameter :: dp = KIND(1.0D0)

   !-----------------------------------------------------------------------
   type IR
      !-----------------------------------------------------------------------
      !!
      !! This contains all the IR-basis objects,
      !! such as sampling points, and basis functions
      !!
      !
      integer :: size
      !! total number of IR basis functions (size of s)
      integer :: ntau
      !! total number of sampling points of imaginary time
      integer :: nfreq_f
      !! total number of sampling Matsubara freqs (Fermionic)
      integer :: nfreq_b
      !! total number of sampling Matsubara freqs (Bosonic)
      integer :: nomega
      !! total number of sampling points of real frequency
      integer :: npoles
      !! total number of DLR poles
      double precision :: beta
      !! inverse temperature
      double precision :: lambda
      !! lambda = 10^{nlambda},
      !! which determines maximum sampling point of real frequency
      double precision :: wmax
      !! maximum real frequency: wmax = lambda / beta
      double precision :: eps
      !! eps = 10^{-ndigit}
      double precision, allocatable :: s(:)
      !! singular values
      double precision, allocatable :: tau(:)
      !! sampling points of imaginary time
      double precision, allocatable :: omega(:)
      !! sampling points of real frequency
      integer(8), allocatable :: freq_f(:)
      !! integer part of sampling Matsubara freqs (Fermion)
      integer(8), allocatable :: freq_b(:)
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
      type(c_ptr) :: tau_smpl_ptr
      !! pointer to the tau sampling points
      type(c_ptr) :: matsu_f_smpl_ptr
      !! pointer to the fermionic frequency sampling points
      type(c_ptr) :: matsu_b_smpl_ptr
      !! pointer to the bosonic frequency sampling points
      type(c_ptr) :: dlr_f_ptr
      !! pointer to the fermionic DLR
      type(c_ptr) :: dlr_b_ptr
      !! pointer to the bosonic DLR
      !-----------------------------------------------------------------------
   end type IR
   !-----------------------------------------------------------------------

   interface evaluate_tau
      module procedure evaluate_tau_zz_impl, evaluate_tau_dd_impl
   end interface evaluate_tau

   interface evaluate_matsubara
      module procedure evaluate_matsubara_zz_impl, evaluate_matsubara_dz_impl
   end interface evaluate_matsubara

   interface fit_tau
      module procedure fit_tau_zz_impl, fit_tau_dd_impl
   end interface fit_tau

   interface fit_matsubara
      module procedure fit_matsubara_zz_impl
   end interface fit_matsubara

   interface ir2dlr
      module procedure ir2dlr_zz_impl, ir2dlr_dd_impl
   end interface ir2dlr

   interface dlr2ir
      module procedure dlr2ir_zz_impl, dlr2ir_dd_impl
   end interface dlr2ir

contains
   function create_logistic_kernel(lambda) result(k_ptr)
      double precision, intent(in) :: lambda
      double precision, TARGET :: lambda_c
      integer(c_int), TARGET :: status_c
      type(c_ptr) :: k_ptr
      lambda_c = lambda
      k_ptr = c_spir_logistic_kernel_new(lambda_c, c_loc(status_c))
   end function create_logistic_kernel

   function create_sve_result(lambda, eps, k_ptr) result(sve_ptr)
      double precision, intent(in) :: lambda
      double precision, intent(in) :: eps

      real(c_double), TARGET :: lambda_c, eps_c
      integer(c_int), TARGET :: status_c

      type(c_ptr), intent(in) :: k_ptr
      type(c_ptr) :: sve_ptr

      lambda_c = lambda
      eps_c = eps

      sve_ptr = c_spir_sve_result_new(k_ptr, eps_c, c_loc(status_c))
      if (status_c /= 0) THEN
         call errore('create_sve_result', 'Error creating SVE result', status_c)
      endif
   end function create_sve_result

   function create_basis(statistics, beta, wmax, k_ptr, sve_ptr) result(basis_ptr)
      integer, intent(in) :: statistics
      double precision, intent(in) :: beta
      double precision, intent(in) :: wmax
      type(c_ptr), intent(in) :: k_ptr, sve_ptr

      integer(c_int), TARGET :: status_c
      type(c_ptr) :: basis_ptr
      integer(c_int) :: statistics_c
      real(c_double) :: beta_c, wmax_c

      statistics_c = statistics
      beta_c = beta
      wmax_c = wmax

      basis_ptr = c_spir_basis_new(statistics_c, beta_c, wmax_c, k_ptr, sve_ptr, c_loc(status_c))
      if (status_c /= 0) THEN
         call errore('create_basis', 'Error creating basis', status_c)
      endif
   end function create_basis

   function get_basis_size(basis_ptr) result(size)
      type(c_ptr), intent(in) :: basis_ptr
      integer(c_int) :: size
      integer(c_int), TARGET :: size_c
      integer(c_int) :: status

      status = c_spir_basis_get_size(basis_ptr, c_loc(size_c))
      if (status /= 0) THEN
         call errore('get_basis_size', 'Error getting basis size', status)
      endif
      size = size_c
   end function get_basis_size

   function basis_get_svals(basis_ptr) result(svals)
      type(c_ptr), intent(in) :: basis_ptr
      double precision, allocatable :: svals(:)
      integer(c_int), TARGET :: nsvals_c
      integer(c_int) :: status
      real(c_double), allocatable, TARGET :: svals_c(:)

      status = c_spir_basis_get_size(basis_ptr, c_loc(nsvals_c))
      if (status /= 0) THEN
         call errore('basis_get_svals', 'Error getting number of singular values', status)
      endif

      allocate(svals_c(nsvals_c))

      status = c_spir_basis_get_svals(basis_ptr, c_loc(svals_c))
      if (status /= 0) THEN
         call errore('basis_get_svals', 'Error getting singular values', status)
      endif

      allocate(svals(nsvals_c))
      svals = svals_c

      deallocate(svals_c)
   end function basis_get_svals

   subroutine create_tau_smpl(basis_ptr, tau, tau_smpl_ptr)
      type(c_ptr), intent(in) :: basis_ptr
      double precision, allocatable, intent(out) :: tau(:)
      type(c_ptr), intent(out) :: tau_smpl_ptr

      integer(c_int), TARGET :: ntau_c
      integer(c_int), TARGET :: status_c
      real(c_double), allocatable, TARGET :: tau_c(:)

      status_c = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau_c))
      if (status_c /= 0) THEN
         call errore('create_tau_smpl', 'Error getting number of tau points', status_c)
      endif

      allocate(tau_c(ntau_c))
      if (allocated(tau)) deallocate(tau)
      allocate(tau(ntau_c))

      status_c = c_spir_basis_get_default_taus(basis_ptr, c_loc(tau_c))
      if (status_c /= 0) THEN
         call errore('create_tau_smpl', 'Error getting tau points', status_c)
      endif
      tau = real(tau_c, KIND=8)

      tau_smpl_ptr = c_spir_tau_sampling_new(basis_ptr, ntau_c, c_loc(tau_c), c_loc(status_c))
      if (status_c /= 0) THEN
         call errore('create_tau_smpl', 'Error creating tau sampling points', status_c)
      endif

      deallocate(tau_c)
   end subroutine create_tau_smpl

   subroutine create_matsu_smpl(basis_ptr, positive_only, matsus, matsu_smpl_ptr)
      type(c_ptr), intent(in) :: basis_ptr
      LOGICAL, intent(in) :: positive_only
      integer(8), allocatable, intent(out) :: matsus(:)
      type(c_ptr), intent(out) :: matsu_smpl_ptr

      integer(c_int), TARGET :: nfreq_c
      integer(c_int), TARGET :: status_c
      integer(c_int64_t), allocatable, TARGET :: matsus_c(:)
      integer(c_int) :: positive_only_c

      positive_only_c = MERGE(1, 0, positive_only)

      status_c = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only_c, c_loc(nfreq_c))
      if (status_c /= 0) THEN
         call errore('create_matsu_smpl', 'Error getting number of fermionic frequencies', status_c)
      endif

      allocate(matsus_c(nfreq_c))

      status_c = c_spir_basis_get_default_matsus(basis_ptr, positive_only_c, c_loc(matsus_c))
      if (status_c /= 0) THEN
         call errore('create_matsu_smpl', 'Error getting frequencies', status_c)
      endif

      if (allocateD(matsus)) deallocate(matsus)
      allocate(matsus(nfreq_c))
      matsus = matsus_c

      ! Create sampling object
      matsu_smpl_ptr = c_spir_matsu_sampling_new(basis_ptr, positive_only_c, nfreq_c, c_loc(matsus_c), c_loc(status_c))
      if (status_c /= 0) THEN
         call errore('create_matsu_smpl', 'Error creating sampling object', status_c)
      endif

      deallocate(matsus_c)
   end subroutine create_matsu_smpl

   function basis_get_ws(basis_ptr) result(ws)
      type(c_ptr), intent(in) :: basis_ptr
      double precision, allocatable :: ws(:)
      integer(c_int), TARGET :: nomega_c
      integer(c_int) :: status
      real(c_double), allocatable, TARGET :: ws_c(:)

      status = c_spir_basis_get_n_default_ws(basis_ptr, c_loc(nomega_c))
      if (status /= 0) THEN
         call errore('basis_get_ws', 'Error getting number of real frequencies', status)
      endif

      allocate(ws_c(nomega_c))

      status = c_spir_basis_get_default_ws(basis_ptr, c_loc(ws_c))
      if (status /= 0) THEN
         call errore('basis_get_ws', 'Error getting real frequencies', status)
      endif

      allocate(ws(nomega_c))
      ws = ws_c

      deallocate(ws_c)
   end function basis_get_ws


   subroutine init_ir(obj, beta, lambda, eps, positive_only)
      !-----------------------------------------------------------------------
      !!
      !! This routine initializes arrays related to the IR-basis objects.
      !! This routine should be called by read_ir or mk_ir_preset.
      !! Do not call in other routines directly.
      !!
      !
      type(IR), INTENT(INOUT) :: obj
      !! contains all the IR-basis objects
      double precision, intent(in) :: beta
      !! inverse temperature
      double precision, intent(in) :: lambda
      !! lambda = 10^{nlambda}
      double precision, intent(in) :: eps
      !! cutoff for the singular value expansion
      LOGICAL, intent(in) :: positive_only
      !! if true, take the Matsubara frequencies
      !! only from the positive region

      double precision :: wmax
      integer(c_int), TARGET :: status_c, npoles_c

      type(c_ptr) :: sve_ptr, basis_f_ptr, basis_b_ptr, k_ptr, dlr_f_ptr, dlr_b_ptr

      wmax = lambda / beta

      k_ptr = create_logistic_kernel(lambda)
      if (.not. c_associated(k_ptr)) then
         call errore('init_ir', 'Kernel is not assigned', 1)
      end if

      sve_ptr = create_sve_result(lambda, eps, k_ptr)
      if (.not. c_associated(sve_ptr)) then
         call errore('init_ir', 'SVE result is not assigned', 1)
      end if

      basis_f_ptr = create_basis(SPIR_STATISTICS_FERMIONIC, beta, wmax, k_ptr, sve_ptr)
      if (.not. c_associated(basis_f_ptr)) then
         call errore('init_ir', 'Fermionic basis is not assigned', 1)
      end if

      basis_b_ptr = create_basis(SPIR_STATISTICS_BOSONIC, beta, wmax, k_ptr, sve_ptr)
      if (.not. c_associated(basis_b_ptr)) then
         call errore('init_ir', 'Bosonic basis is not assigned', 1)
      end if

      ! Create DLR objects
      dlr_f_ptr = c_spir_dlr_new(basis_f_ptr, c_loc(status_c))
      if (status_c /= 0 .or. .not. c_associated(dlr_f_ptr)) then
         call errore('init_ir', 'Error creating fermionic DLR', status_c)
      end if

      dlr_b_ptr = c_spir_dlr_new(basis_b_ptr, c_loc(status_c))
      if (status_c /= 0 .or. .not. c_associated(dlr_b_ptr)) then
         call errore('init_ir', 'Error creating bosonic DLR', status_c)
      end if

      ! Get number of poles
      status_c = c_spir_dlr_get_npoles(dlr_f_ptr, c_loc(npoles_c))
      if (status_c /= 0) then
         call errore('init_ir', 'Error getting number of poles', status_c)
      end if

      obj%basis_f_ptr = basis_f_ptr
      obj%basis_b_ptr = basis_b_ptr
      obj%sve_ptr = sve_ptr
      obj%k_ptr = k_ptr
      obj%dlr_f_ptr = dlr_f_ptr
      obj%dlr_b_ptr = dlr_b_ptr
      obj%npoles = npoles_c

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
      obj%eps = eps

   end subroutine


   subroutine finalize_ir(obj)
      !-----------------------------------------------------------------------
      !!
      !! This routine deallocates IR-basis objects contained in obj
      !!
      !
      type(IR) :: obj
      !! contains all the IR-basis objects
      integer :: ierr
      !! Error status
      !
      ! Deallocate all member variables
      deallocate(obj%s, STAT = ierr)
      if (ierr /= 0) call errore('finalize_ir', 'Error deallocating IR%s', 1)
      deallocate(obj%tau, STAT = ierr)
      if (ierr /= 0) call errore('finalize_ir', 'Error deallocating IR%tau', 1)
      deallocate(obj%omega, STAT = ierr)
      if (ierr /= 0) call errore('finalize_ir', 'Error deallocating IR%omega', 1)
      deallocate(obj%freq_f, STAT = ierr)
      if (ierr /= 0) call errore('finalize_ir', 'Error deallocating IR%freq_f', 1)
      deallocate(obj%freq_b, STAT = ierr)
      if (ierr /= 0) call errore('finalize_ir', 'Error deallocating IR%freq_b', 1)
      !-----------------------------------------------------------------------

      if (c_associated(obj%basis_f_ptr)) then
         call c_spir_basis_release(obj%basis_f_ptr)
      end if
      if (c_associated(obj%basis_b_ptr)) then
         call c_spir_basis_release(obj%basis_b_ptr)
      end if
      if (c_associated(obj%sve_ptr)) then
         call c_spir_sve_result_release(obj%sve_ptr)
      end if
      if (c_associated(obj%k_ptr)) then
         call c_spir_kernel_release(obj%k_ptr)
      end if
      if (c_associated(obj%tau_smpl_ptr)) then
         call c_spir_sampling_release(obj%tau_smpl_ptr)
      end if
      if (c_associated(obj%matsu_f_smpl_ptr)) then
         call c_spir_sampling_release(obj%matsu_f_smpl_ptr)
      end if
      if (c_associated(obj%matsu_b_smpl_ptr)) then
         call c_spir_sampling_release(obj%matsu_b_smpl_ptr)
      end if
      if (c_associated(obj%dlr_f_ptr)) then
         call c_spir_basis_release(obj%dlr_f_ptr)
      end if
      if (c_associated(obj%dlr_b_ptr)) then
         call c_spir_basis_release(obj%dlr_b_ptr)
      end if
   end subroutine finalize_ir

   subroutine errore(routine, msg, ierr)
      !-----------------------------------------------------------------------
      !!
      !! This routine handles error messages and program termination
      !!
      !
      character(*), intent(in) :: routine, msg
      integer, intent(in) :: ierr
      !
      ! Print error message with asterisk border
      WRITE(UNIT = 0, FMT = '(/,1X,78("*"))')
      WRITE(UNIT = 0, FMT = '(5X,"from ",A," : error #",I10)') routine, ierr
      WRITE(UNIT = 0, FMT = '(5X,A)') msg
      WRITE(UNIT = 0, FMT = '(1X,78("*"),/)')
      !
      STOP
      !-----------------------------------------------------------------------
   end subroutine errore


   subroutine flatten_dd(x, flat)
      double precision, intent(in) :: x(..) ! Arbitrary rank
      double precision, allocatable, intent(out) :: flat(:) ! 1D array

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

   subroutine flatten_zz(x, flat)
      complex(kind=dp), intent(in) :: x(..) ! Arbitrary rank
      complex(kind=dp), allocatable, intent(out) :: flat(:) ! 1D array

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
      complex(kind=dp), intent(in) :: x(..) ! Arbitrary rank
      double precision, allocatable, intent(out) :: flat(:) ! 1D array

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
      complex(kind=dp), intent(in) :: flat(:) ! 1D array
      complex(kind=dp), intent(out) :: x(..) ! Arbitrary rank

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
      double precision, intent(in) :: flat(:) ! 1D array
      double precision, intent(out) :: x(..) ! Arbitrary rank

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
      double precision, intent(in) :: flat(:) ! 1D array
      complex(kind=dp), intent(out) :: x(..) ! Arbitrary rank

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
      integer, intent(in) :: target_dim
      integer(c_int), intent(in) :: input_dims(:)
      integer(c_int), intent(in) :: output_dims(:)
      LOGICAL :: is_valid

      integer :: i

      if (size(input_dims) /= size(output_dims)) then
         write(*, *) "input_dims and output_dims have different sizes"
         stop
      end if

      do i = 1, size(input_dims)
         if (i == target_dim) then
            cycle
         end if
         if (input_dims(i) /= output_dims(i)) then
            print *, "input_dims(", i, ")", input_dims(i), "output_dims(", i, ")", output_dims(i)
            is_valid = .false.
            return
         end if
      end do
      is_valid = .true.
   end function check_output_dims

   subroutine evaluate_tau_zz_impl(obj, target_dim, arr, res)
      type(IR), intent(in) :: obj
      integer, intent(in) :: target_dim
      complex(kind=dp), intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('evaluate_tau_zz_impl', 'Target dimension is out of range', 1)
      end if

      if (input_dims_c(target_dim) /= obj%size) then
         call errore('evaluate_tau_zz_impl', 'Target dimension is not the same as the basis size', 1)
      end if

      if (output_dims_c(target_dim) /= size(obj%tau)) then
         call errore('evaluate_tau_zz_impl', 'Target dimension is not the same as the number of tau sampling points', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('evaluate_tau_zz_impl', &
          'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      target_dim_c = target_dim - 1
      if (obj%positive_only) THEN
         block
            real(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
            call flatten_zd(arr, arr_c)

            allocate(res_c(product(output_dims_c)))

            status_c = c_spir_sampling_eval_dd(obj%tau_smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
               ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

            if (status_c /= 0) then
               call errore('evaluate_tau_zz_impl', 'Error evaluating on tau sampling points', status_c)
            end if

            call unflatten_dz(res_c, res)
            deallocate(arr_c, res_c)
         end block
      ELSE
         block
            COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
            call flatten_zz(arr, arr_c)

            allocate(res_c(product(output_dims_c)))

            status_c = c_spir_sampling_eval_zz(obj%tau_smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
               ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

            if (status_c /= 0) then
               call errore('evaluate_tau_zz_impl', 'Error evaluating on tau sampling points', status_c)
            end if

            call unflatten_zz(res_c, res)
            deallocate(arr_c, res_c)
         end block
      end if
   end subroutine evaluate_tau_zz_impl


   subroutine evaluate_tau_dd_impl(obj, target_dim, arr, res)
      type(IR), intent(in) :: obj
      integer, intent(in) :: target_dim
      double precision, intent(in) :: arr(..)
      double precision, intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      real(c_double), allocatable, TARGET :: arr_c(:), res_c(:)

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in output_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('evaluate_tau_dd_impl', 'Target dimension is out of range', 1)
      end if

      if (input_dims_c(target_dim) /= obj%size) then
         call errore('evaluate_tau_zz_impl', 'Target dimension is not the same as the basis size', 1)
      end if

      if (output_dims_c(target_dim) /= size(obj%tau)) then
         call errore('evaluate_tau_zz_impl', 'Target dimension is not the same as the number of tau sampling points', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('evaluate_tau_zz_impl', &
            'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      call flatten_dd(arr, arr_c)

      target_dim_c = target_dim - 1
      status_c = c_spir_sampling_eval_dd(obj%tau_smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
         ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

      if (status_c /= 0) then
         call errore('evaluate_tau_dd_impl', 'Error evaluating on tau sampling points', status_c)
      end if

      call unflatten_dd(res_c, res)
      deallocate(arr_c, res_c)
   end subroutine evaluate_tau_dd_impl

   subroutine evaluate_matsubara_zz_internal(smpl_ptr, target_dim, arr, res)
      type(c_ptr), intent(in) :: smpl_ptr
      integer, intent(in) :: target_dim
      complex(kind=dp), intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('evaluate_matsubara_zz_internal', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('evaluate_matsubara_zz_internal', &
            'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if


      target_dim_c = target_dim - 1
      block
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         call flatten_zz(arr, arr_c)

         allocate(res_c(product(output_dims_c)))

         status_c = c_spir_sampling_eval_zz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            call errore('evaluate_matsubara_zz_internal', 'Error evaluating on Matsubara frequencies', status_c)
         end if

         call unflatten_zz(res_c, res)
         deallocate(arr_c, res_c)
      end block
   end subroutine evaluate_matsubara_zz_internal

   subroutine evaluate_matsubara_dz_internal(smpl_ptr, target_dim, arr, res)
      type(c_ptr), intent(in) :: smpl_ptr
      integer, intent(in) :: target_dim
      double precision, intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      real(c_double), allocatable, TARGET :: arr_c(:), res_c(:)

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in output_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('evaluate_matsubara_dz_internal', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('evaluate_matsubara_dz_internal', &
            'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      allocate(res_c(product(output_dims_c)))

      call flatten_dd(arr, arr_c)

      target_dim_c = target_dim - 1
      status_c = c_spir_sampling_eval_dz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
         ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

      if (status_c /= 0) then
         call errore('evaluate_matsubara_dz_internal', 'Error evaluating on Matsubara frequencies', status_c)
      end if

      call unflatten_dz(res_c, res)
      deallocate(arr_c, res_c)
   end subroutine evaluate_matsubara_dz_internal

   subroutine evaluate_matsubara_zz_impl(obj, statistics, target_dim, arr, res)
      type(IR), intent(in) :: obj
      integer, intent(in) :: statistics
      integer, intent(in) :: target_dim
      complex(kind=dp), intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)

      if (statistics == SPIR_STATISTICS_FERMIONIC) then
         call evaluate_matsubara_zz_internal(obj%matsu_f_smpl_ptr, target_dim, arr, res)
      ELSE if (statistics == SPIR_STATISTICS_BOSONIC) then
         call evaluate_matsubara_zz_internal(obj%matsu_b_smpl_ptr, target_dim, arr, res)
      ELSE
         call errore('evaluate_matsubara_zz_impl', 'Invalid statistics', 1)
      end if
   end subroutine evaluate_matsubara_zz_impl


   subroutine evaluate_matsubara_dz_impl(obj, statistics, target_dim, arr, res)
      type(IR), intent(in) :: obj
      integer, intent(in) :: statistics
      integer, intent(in) :: target_dim
      double precision, intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)

      if (statistics == SPIR_STATISTICS_FERMIONIC) then
         call evaluate_matsubara_dz_internal(obj%matsu_f_smpl_ptr, target_dim, arr, res)
      ELSE if (statistics == SPIR_STATISTICS_BOSONIC) then
         call evaluate_matsubara_dz_internal(obj%matsu_b_smpl_ptr, target_dim, arr, res)
      ELSE
         call errore('evaluate_matsubara_dz', 'Invalid statistics', 1)
      end if
   end subroutine evaluate_matsubara_dz_impl

   subroutine fit_tau_zz_impl(smpl_ptr, target_dim, arr, res)
      type(c_ptr), intent(in) :: smpl_ptr
      integer, intent(in) :: target_dim
      complex(kind=dp), intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('fit_tau_zz_impl', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('fit_tau_zz_impl', &
            'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      target_dim_c = target_dim - 1
      block
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         call flatten_zz(arr, arr_c)

         allocate(res_c(product(output_dims_c)))

         status_c = c_spir_sampling_fit_zz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            call errore('fit_tau_zz_impl', 'Error fitting on tau sampling points', status_c)
         end if

         call unflatten_zz(res_c, res)
         deallocate(arr_c, res_c)
      end block
   end subroutine fit_tau_zz_impl

   subroutine fit_tau_dd_impl(smpl_ptr, target_dim, arr, res)
      type(c_ptr), intent(in) :: smpl_ptr
      integer, intent(in) :: target_dim
      double precision, intent(in) :: arr(..)
      double precision, intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      real(c_double), allocatable, TARGET :: arr_c(:), res_c(:)

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in output_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('fit_tau_dd_impl', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('fit_tau_dd_impl', &
            'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      call flatten_dd(arr, arr_c)

      target_dim_c = target_dim - 1
      status_c = c_spir_sampling_fit_dd(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
         ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

      if (status_c /= 0) then
         call errore('fit_tau_dd_impl', 'Error fitting on tau sampling points', status_c)
      end if

      call unflatten_dd(res_c, res)
      deallocate(arr_c, res_c)
   end subroutine fit_tau_dd_impl

   subroutine fit_matsubara_zz_internal(smpl_ptr, target_dim, arr, res, positive_only)
      type(c_ptr), intent(in) :: smpl_ptr
      integer, intent(in) :: target_dim
      complex(kind=dp), intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)
      LOGICAL, intent(in), OPTIONAL :: positive_only

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('fit_matsubara_zz_internal', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('fit_matsubara_zz_internal', &
            'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      target_dim_c = target_dim - 1
      block
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         call flatten_zz(arr, arr_c)

         allocate(res_c(product(output_dims_c)))

         status_c = c_spir_sampling_fit_zz(smpl_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            call errore('fit_matsubara_zz_internal', 'Error fitting on Matsubara frequencies', status_c)
         end if

         call unflatten_zz(res_c, res)
         deallocate(arr_c, res_c)
      end block

   end subroutine fit_matsubara_zz_internal

   subroutine fit_matsubara_zz_impl(obj, statistics, target_dim, arr, res)
      type(IR), intent(in) :: obj
      integer, intent(in) :: statistics
      integer, intent(in) :: target_dim
      complex(kind=dp), intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)

      if (statistics == SPIR_STATISTICS_FERMIONIC) then
         call fit_matsubara_zz_internal(obj%matsu_f_smpl_ptr, target_dim, arr, res, obj%positive_only)
      ELSE if (statistics == SPIR_STATISTICS_BOSONIC) then
         call fit_matsubara_zz_internal(obj%matsu_b_smpl_ptr, target_dim, arr, res, obj%positive_only)
      ELSE
         call errore('fit_matsubara_zz_impl', 'Invalid statistics', 1)
      end if
   end subroutine fit_matsubara_zz_impl


   subroutine ir2dlr_zz_impl(obj, target_dim, arr, res)
      type(IR), intent(in) :: obj
      integer, intent(in) :: target_dim
      complex(kind=dp), intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('ir2dlr_zz_impl', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('ir2dlr_zz_impl', &
            'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      target_dim_c = target_dim - 1
      block
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         call flatten_zz(arr, arr_c)

         status_c = c_spir_ir2dlr_zz(obj%dlr_f_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            call errore('ir2dlr_zz_impl', 'Error converting IR to DLR', status_c)
         end if

         call unflatten_zz(res_c, res)
      end block
   end subroutine ir2dlr_zz_impl

   subroutine ir2dlr_dd_impl(obj, target_dim, arr, res)
      type(IR), intent(in) :: obj
      integer, intent(in) :: target_dim
      double precision, intent(in) :: arr(..)
      double precision, intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('ir2dlr_dd_impl', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('ir2dlr_dd_impl', &
             'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      target_dim_c = target_dim - 1
      block
         real(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         call flatten_dd(arr, arr_c)

         status_c = c_spir_ir2dlr_dd(obj%dlr_f_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            call errore('ir2dlr_dd_impl', 'Error converting IR to DLR', status_c)
         end if

         call unflatten_dd(res_c, res)
      end block
   end subroutine ir2dlr_dd_impl

   subroutine dlr2ir_zz_impl(obj, target_dim, arr, res)
      type(IR), intent(in) :: obj
      integer, intent(in) :: target_dim
      complex(kind=dp), intent(in) :: arr(..)
      complex(kind=dp), intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('dlr2ir_zz_impl', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('dlr2ir_zz_impl', &
         'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      if (size(arr, target_dim) /= obj%npoles) then
         call errore('dlr2ir_zz_impl', 'Input dimension is not the same as the number of poles', 1)
      end if

      if (size(res, target_dim) /= obj%size) then
         call errore('dlr2ir_zz_impl', 'Output dimension is not the same as the size of the IR', 1)
      end if

      target_dim_c = target_dim - 1
      block
         COMPLEX(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         call flatten_zz(arr, arr_c)

         allocate(res_c(product(output_dims_c)))

         status_c = c_spir_dlr2ir_zz(obj%dlr_f_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            call errore('dlr2ir_zz_impl', 'Error converting DLR to IR', status_c)
         end if

         call unflatten_zz(res_c, res)
         deallocate(arr_c, res_c)
      end block
   end subroutine dlr2ir_zz_impl

   subroutine dlr2ir_dd_impl(obj, target_dim, arr, res)
      type(IR), intent(in) :: obj
      integer, intent(in) :: target_dim
      double precision, intent(in) :: arr(..)
      double precision, intent(out) :: res(..)

      integer(c_int) :: ndim_c, target_dim_c
      integer(c_int), allocatable, TARGET :: input_dims_c(:), output_dims_c(:)
      integer(c_int) :: status_c

      input_dims_c = shape(arr)
      output_dims_c = shape(res)
      ndim_c = size(input_dims_c)

      ! check target_dim is in input_dims_c
      if (target_dim <= 0 .or. target_dim > ndim_c) then
         call errore('dlr2ir_dd_impl', 'Target dimension is out of range', 1)
      end if

      if (.not. check_output_dims(target_dim, input_dims_c, output_dims_c)) then
         call errore('dlr2ir_dd_impl', &
             'Output dimensions are not the same as the input dimensions except for the TARGET dimension', 1)
      end if

      if (size(arr, target_dim) /= obj%npoles) then
         call errore('dlr2ir_dd_impl', 'Input dimension is not the same as the number of poles', 1)
      end if

      if (size(res, target_dim) /= obj%size) then
         call errore('dlr2ir_dd_impl', 'Output dimension is not the same as the size of the IR', 1)
      end if

      target_dim_c = target_dim - 1
      block
         real(c_double), allocatable, TARGET :: arr_c(:), res_c(:)
         call flatten_dd(arr, arr_c)

         allocate(res_c(product(output_dims_c)))

         status_c = c_spir_dlr2ir_dd(obj%dlr_f_ptr, SPIR_ORDER_COLUMN_MAJOR, &
            ndim_c, c_loc(input_dims_c), target_dim_c, c_loc(arr_c), c_loc(res_c))

         if (status_c /= 0) then
            call errore('dlr2ir_dd_impl', 'Error converting DLR to IR', status_c)
         end if

         call unflatten_dd(res_c, res)

         deallocate(arr_c, res_c)
      end block
   end subroutine dlr2ir_dd_impl

   function eval_u_tau(obj, tau) result(res)
      type(IR), intent(in) :: obj
      double precision, intent(in) :: tau

      double precision, allocatable :: res(:)
      real(c_double), allocatable, target :: res_c(:)
      integer(c_int), target :: status_c

      type(c_ptr) :: u_tau_ptr

      u_tau_ptr = c_spir_basis_get_u(obj%basis_f_ptr, c_loc(status_c))
      if (.not. c_associated(u_tau_ptr)) then
         call errore('eval_u_tau', 'Error getting u_tau pointer', status_c)
      end if

      allocate(res_c(obj%size))
      status_c = c_spir_funcs_eval(u_tau_ptr, tau, c_loc(res_c))
      if (status_c /= 0) then
         call errore('eval_u_tau', 'Error evaluating u_tau', status_c)
      end if

      res = real(res_c, KIND=DP)

      deallocate(res_c)
      call c_spir_funcs_release(u_tau_ptr)
   end function eval_u_tau

end module sparseir_ext