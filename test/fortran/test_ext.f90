! Simple test program for SparseIR Fortran bindings
program test_ext
   use sparseir
   use sparseir_ext
   implicit none

   type(IR) :: irobj

   DOUBLE PRECISION, parameter :: beta = 1e+2
   DOUBLE PRECISION, parameter :: lambda = 1.0
   DOUBLE PRECISION, parameter :: eps = 1.0e-10
   LOGICAL, parameter :: positive_only = .true.

   call init_ir(irobj, beta, lambda, eps, positive_only)

   contains
end program test_ext