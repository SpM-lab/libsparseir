#include "sparseir/gemm.hpp"

#ifdef SPARSEIR_USE_BLAS
namespace sparseir {
#endif // SPARSEIR_USE_BLAS

#ifdef SPARSEIR_USE_EXTERN_FBLAS_PTR
    #include "gemm_external.impl"
#else
    #include "gemm_link.impl"
#endif

#ifdef SPARSEIR_USE_BLAS
} // namespace sparseir
#endif // SPARSEIR_USE_BLAS
