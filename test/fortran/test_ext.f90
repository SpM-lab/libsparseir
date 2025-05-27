! Simple test program for SparseIR Fortran bindings
program test_ext
   use sparseir
   use sparseir_ext
   implicit none

   call test_create_sve_result()

   contains
   subroutine test_create_sve_result()
      double precision :: lambda, eps
      type(c_ptr) :: sve_ptr

      lambda = 1.0
      eps = 1.0e-10
      sve_ptr = create_sve_result(lambda, eps)

      if (.not. c_associated(sve_ptr)) then
         print*, "Error: SVE result is not assigned"
         stop
      else
         print*, "SVE result is assigned"
      end if

   end subroutine test_create_sve_result
end program test_ext