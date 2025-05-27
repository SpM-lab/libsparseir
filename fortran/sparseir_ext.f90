! Extends the SparseIR library with additional functionality
module sparseir_ext
  use, intrinsic :: iso_c_binding
  use sparseir
  implicit none
  
  type(c_ptr) :: sve_ptr

  contains
  FUNCTION create_sve_result(lambda, eps) result(sve_ptr)
    DOUBLE PRECISION, INTENT(IN) :: lambda
    DOUBLE PRECISION, INTENT(IN) :: eps

    REAL(c_double), target :: lambda_c, eps_c
    INTEGER(c_int), target :: status_c

    type(c_ptr) :: k_ptr
    type(c_ptr) :: sve_ptr

    lambda_c = lambda
    eps_c = eps

    k_ptr = c_spir_logistic_kernel_new(lambda_c, c_loc(status_c))
    IF (status_c /= 0) THEN
       PRINT*, "Error creating kernel"
       STOP
    ENDIF

    sve_ptr = c_spir_sve_result_new(k_ptr, eps_c, c_loc(status_c))
    IF (status_c /= 0) THEN
       PRINT*, "Error creating SVE result"
       STOP
    ENDIF
  end function create_sve_result

  SUBROUTINE mod_create_sve_result(lambda, eps)
    DOUBLE PRECISION, INTENT(IN) :: lambda
    DOUBLE PRECISION, INTENT(IN) :: eps

    sve_ptr = create_sve_result(lambda, eps)
  END SUBROUTINE mod_create_sve_result

end module sparseir_ext