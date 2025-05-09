interface is_initialized
  ! Check if kernel is initialized
  module function kernel_is_initialized(this) result(initialized)
    class(spir_kernel), intent(in) :: this
    logical :: initialized
  end function

  ! Check if sve_result is initialized
  module function sve_result_is_initialized(this) result(initialized)
    class(spir_sve_result), intent(in) :: this
    logical :: initialized
  end function

end interface

interface clone
  ! Clone the kernel object (create a copy)
  module function kernel_clone(this) result(copy)
    class(spir_kernel), intent(in) :: this
    type(spir_kernel) :: copy
  end function

    ! Clone the sve_result object (create a copy)
  module function sve_result_clone(this) result(copy)
    class(spir_sve_result), intent(in) :: this
    type(spir_sve_result) :: copy
  end function

  end interface

interface assign
  ! Assignment operator implementation
  module subroutine kernel_assign(lhs, rhs)
    class(spir_kernel), intent(inout) :: lhs
    class(spir_kernel), intent(in) :: rhs
  end subroutine

  ! Assignment operator implementation
  module subroutine sve_result_assign(lhs, rhs)
    class(spir_sve_result), intent(inout) :: lhs
    class(spir_sve_result), intent(in) :: rhs
  end subroutine

end interface

interface finalize
  ! Finalizer for kernel
  module subroutine kernel_finalize(this)
    type(spir_kernel), intent(inout) :: this
  end subroutine

  ! Finalizer for sve_result
  module subroutine sve_result_finalize(this)
    type(spir_sve_result), intent(inout) :: this
  end subroutine

end interface

interface is_assigned
  ! Check if kernel's shared_ptr is assigned
  ! (contains a valid C++ object)
  module function kernel_is_assigned(kernel) result(assigned)
    type(spir_kernel), intent(in) :: kernel
    logical :: assigned
  end function

  ! Check if sve_result's shared_ptr is assigned
  ! (contains a valid C++ object)
  module function sve_result_is_assigned(sve_result) result(assigned)
    type(spir_sve_result), intent(in) :: sve_result
    logical :: assigned
  end function

end interface

