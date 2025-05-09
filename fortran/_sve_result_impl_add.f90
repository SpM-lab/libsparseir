  module procedure spir_sve_result_new
    integer(c_int) :: status

    status = c_spir_sve_result_new(result%ptr, k%ptr, epsilon)
    if (status /= 0) then
      print *, "Warning: C function returned error status =", status
      result%ptr = c_null_ptr
    end if
  end procedure
