#include "sparseir/gemm.hpp"

namespace sparseir {

#ifdef SPARSEIR_USE_EXTERN_FBLAS_PTR
    #include "gemm_external.impl"
#else
    #include "gemm_link.impl"
#endif

} // namespace sparseir
