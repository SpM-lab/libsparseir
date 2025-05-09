    ! Check if sve_result is initialized
    module function sve_result_is_initialized(this) result(initialized)
      class(spir_sve_result), intent(in) :: this
      logical :: initialized
    end function

    ! Clone the sve_result object (create a copy)
    module function sve_result_clone(this) result(copy)
      class(spir_sve_result), intent(in) :: this
      type(spir_sve_result) :: copy
    end function

    ! Assignment operator implementation
    module subroutine sve_result_assign(lhs, rhs)
      class(spir_sve_result), intent(inout) :: lhs
      class(spir_sve_result), intent(in) :: rhs
    end subroutine

    ! Finalizer for sve_result
    module subroutine sve_result_finalize(this)
      type(spir_sve_result), intent(inout) :: this
    end subroutine
