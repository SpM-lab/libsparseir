#include "sparseir/gemm.hpp"

#include <cstddef>
#include <stdexcept>
#include <dlfcn.h>

namespace sparseir {

// Fortran dgemm function pointer types (both ILP64 and LP64)
using dgemm_fptr_lp64 = void(*)(const char*, const char*, const int*,
                                const int*, const int*, const double*,
                                const double*, const int*, const double*,
                                const int*, const double*, double*, const int*);

using zgemm_fptr_lp64 = void(*)(const char*, const char*, const int*,
                                const int*, const int*, const void*,
                                const void*, const int*, const void*,
                                const int*, const void*, void*, const int*);

using dgemm_fptr_ilp64 = void(*)(const char*, const char*, const long long*,
                                  const long long*, const long long*, const double*,
                                  const double*, const long long*, const double*,
                                  const long long*, const double*, double*, const long long*);

using zgemm_fptr_ilp64 = void(*)(const char*, const char*, const long long*,
                                  const long long*, const long long*, const void*,
                                  const void*, const long long*, const void*,
                                  const long long*, const void*, void*, const long long*);

// Storage for registered function pointers
static dgemm_fptr_lp64 registered_dgemm_lp64 = nullptr;
static zgemm_fptr_lp64 registered_zgemm_lp64 = nullptr;
static dgemm_fptr_ilp64 registered_dgemm_ilp64 = nullptr;
static zgemm_fptr_ilp64 registered_zgemm_ilp64 = nullptr;
static bool symbols_resolved = false;
static bool use_ilp64 = false;

// Function to resolve BLAS symbols at runtime
template<typename FuncPtr>
FuncPtr resolve_blas_symbol(const char* patterns[], size_t count) {
    // Try RTLD_DEFAULT first (searches already-loaded libraries)
    void* handle = RTLD_DEFAULT;
    
    for (size_t i = 0; i < count; ++i) {
        void* sym = dlsym(handle, patterns[i]);
        if (sym) {
            return reinterpret_cast<FuncPtr>(sym);
        }
    }
    
    // If not found, try dlopen(NULL) to search main program and all loaded libraries
    handle = dlopen(NULL, RTLD_LAZY);
    if (handle) {
        for (size_t i = 0; i < count; ++i) {
            void* sym = dlsym(handle, patterns[i]);
            if (sym) {
                dlclose(handle);
                return reinterpret_cast<FuncPtr>(sym);
            }
        }
        dlclose(handle);
    }
    
    return nullptr;
}

// Initialize BLAS symbols once - try ILP64 first, then fall back to LP64
static void init_blas_symbols() {
    if (symbols_resolved) {
        return;
    }
    
#ifdef SPARSEIR_USE_EXTERN_FBLAS_PTR
    // When using external function pointers, check if they have been registered
    if ((!registered_dgemm_lp64 && !registered_dgemm_ilp64) ||
        (!registered_zgemm_lp64 && !registered_zgemm_ilp64)) {
        throw std::runtime_error(
            "BLAS function pointers not registered - call spir_register_dgemm_zgemm_lp64() "
            "or spir_register_dgemm_zgemm_ilp64() before using BLAS functions");
    }
    // Determine which interface is being used
    use_ilp64 = (registered_dgemm_ilp64 != nullptr && registered_zgemm_ilp64 != nullptr);
    symbols_resolved = true;
    return;
#else
    // First, try ILP64 symbol patterns
    const char* dgemm_patterns_ilp64[] = {
        "dgemm64_", "dgemm64", "DGEMM64_", "DGEMM64"
    };
    const char* zgemm_patterns_ilp64[] = {
        "zgemm64_", "zgemm64", "ZGEMM64_", "ZGEMM64"
    };
    
    registered_dgemm_ilp64 = resolve_blas_symbol<dgemm_fptr_ilp64>(
        dgemm_patterns_ilp64, sizeof(dgemm_patterns_ilp64) / sizeof(dgemm_patterns_ilp64[0]));
    registered_zgemm_ilp64 = resolve_blas_symbol<zgemm_fptr_ilp64>(
        zgemm_patterns_ilp64, sizeof(zgemm_patterns_ilp64) / sizeof(zgemm_patterns_ilp64[0]));
    
    // If ILP64 found, use ILP64
    if (registered_dgemm_ilp64 && registered_zgemm_ilp64) {
        use_ilp64 = true;
        symbols_resolved = true;
        return;
    }
    
    // Fall back to LP64 symbol patterns
    const char* dgemm_patterns_lp64[] = {
        "dgemm_", "dgemm", "DGEMM_", "DGEMM"
    };
    const char* zgemm_patterns_lp64[] = {
        "zgemm_", "zgemm", "ZGEMM_", "ZGEMM"
    };
    
    registered_dgemm_lp64 = resolve_blas_symbol<dgemm_fptr_lp64>(
        dgemm_patterns_lp64, sizeof(dgemm_patterns_lp64) / sizeof(dgemm_patterns_lp64[0]));
    registered_zgemm_lp64 = resolve_blas_symbol<zgemm_fptr_lp64>(
        zgemm_patterns_lp64, sizeof(zgemm_patterns_lp64) / sizeof(zgemm_patterns_lp64[0]));
    
    use_ilp64 = false;
    symbols_resolved = true;
#endif
}

extern "C" {

// Register function pointers from outside (for SPARSEIR_USE_EXTERN_FBLAS_PTR)
// Register both dgemm and zgemm at the same time (LP64)
void spir_register_dgemm_zgemm_lp64(void* dgemm_fn, void* zgemm_fn) {
    registered_dgemm_lp64 = reinterpret_cast<dgemm_fptr_lp64>(dgemm_fn);
    registered_zgemm_lp64 = reinterpret_cast<zgemm_fptr_lp64>(zgemm_fn);
    use_ilp64 = false;
    symbols_resolved = true;
}

// Register function pointers from outside (for SPARSEIR_USE_EXTERN_FBLAS_PTR)
// Register both dgemm and zgemm at the same time (ILP64)
void spir_register_dgemm_zgemm_ilp64(void* dgemm_fn, void* zgemm_fn) {
    registered_dgemm_ilp64 = reinterpret_cast<dgemm_fptr_ilp64>(dgemm_fn);
    registered_zgemm_ilp64 = reinterpret_cast<zgemm_fptr_ilp64>(zgemm_fn);
    use_ilp64 = true;
    symbols_resolved = true;
}

} // extern "C"

// Implementation of my_dgemm - Fortran BLAS interface (LP64)
void my_dgemm(const char* transa, const char* transb,
              const int* m, const int* n, const int* k,
              const double* alpha, const double* a, const int* lda,
              const double* b, const int* ldb, const double* beta,
              double* c, const int* ldc)
{
    // Initialize symbols if not already done
    init_blas_symbols();
    
    if (use_ilp64) {
        // Convert LP64 arguments to ILP64
        if (!registered_dgemm_ilp64) {
            throw std::runtime_error("dgemm (ILP64) symbol not found - BLAS library may not be linked correctly");
        }
        const long long m_ll = static_cast<long long>(*m), n_ll = static_cast<long long>(*n), k_ll = static_cast<long long>(*k);
        const long long lda_ll = static_cast<long long>(*lda), ldb_ll = static_cast<long long>(*ldb), ldc_ll = static_cast<long long>(*ldc);
        registered_dgemm_ilp64(transa, transb, &m_ll, &n_ll, &k_ll, alpha, a, &lda_ll, b, &ldb_ll, beta, c, &ldc_ll);
    } else {
        // Use LP64 directly
        if (!registered_dgemm_lp64) {
            throw std::runtime_error("dgemm (LP64) symbol not found - BLAS library may not be linked correctly");
        }
        registered_dgemm_lp64(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

// Implementation of my_zgemm - Fortran BLAS interface (LP64)
void my_zgemm(const char* transa, const char* transb,
              const int* m, const int* n, const int* k,
              const void* alpha, const void* a, const int* lda,
              const void* b, const int* ldb, const void* beta,
              void* c, const int* ldc)
{
    // Initialize symbols if not already done
    init_blas_symbols();
    
    if (use_ilp64) {
        // Convert LP64 arguments to ILP64
        if (!registered_zgemm_ilp64) {
            throw std::runtime_error("zgemm (ILP64) symbol not found - BLAS library may not be linked correctly");
        }
        const long long m_ll = static_cast<long long>(*m), n_ll = static_cast<long long>(*n), k_ll = static_cast<long long>(*k);
        const long long lda_ll = static_cast<long long>(*lda), ldb_ll = static_cast<long long>(*ldb), ldc_ll = static_cast<long long>(*ldc);
        registered_zgemm_ilp64(transa, transb, &m_ll, &n_ll, &k_ll, alpha, a, &lda_ll, b, &ldb_ll, beta, c, &ldc_ll);
    } else {
        // Use LP64 directly
        if (!registered_zgemm_lp64) {
            throw std::runtime_error("zgemm (LP64) symbol not found - BLAS library may not be linked correctly");
        }
        registered_zgemm_lp64(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

// Implementation of my_dgemm64 - Fortran BLAS interface (ILP64)
void my_dgemm64(const char* transa, const char* transb,
                const long long* m, const long long* n, const long long* k,
                const double* alpha, const double* a, const long long* lda,
                const double* b, const long long* ldb, const double* beta,
                double* c, const long long* ldc)
{
    // Initialize symbols if not already done
    init_blas_symbols();
    
    if (use_ilp64) {
        // Use ILP64 directly
        if (!registered_dgemm_ilp64) {
            throw std::runtime_error("dgemm64 (ILP64) symbol not found - BLAS library may not be linked correctly");
        }
        registered_dgemm_ilp64(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        // Convert ILP64 arguments to LP64
        if (!registered_dgemm_lp64) {
            throw std::runtime_error("dgemm (LP64) symbol not found - BLAS library may not be linked correctly");
        }
        const int m_int = static_cast<int>(*m), n_int = static_cast<int>(*n), k_int = static_cast<int>(*k);
        const int lda_int = static_cast<int>(*lda), ldb_int = static_cast<int>(*ldb), ldc_int = static_cast<int>(*ldc);
        registered_dgemm_lp64(transa, transb, &m_int, &n_int, &k_int, alpha, a, &lda_int, b, &ldb_int, beta, c, &ldc_int);
    }
}

// Implementation of my_zgemm64 - Fortran BLAS interface (ILP64)
void my_zgemm64(const char* transa, const char* transb,
                const long long* m, const long long* n, const long long* k,
                const void* alpha, const void* a, const long long* lda,
                const void* b, const long long* ldb, const void* beta,
                void* c, const long long* ldc)
{
    // Initialize symbols if not already done
    init_blas_symbols();
    
    if (use_ilp64) {
        // Use ILP64 directly
        if (!registered_zgemm_ilp64) {
            throw std::runtime_error("zgemm64 (ILP64) symbol not found - BLAS library may not be linked correctly");
        }
        registered_zgemm_ilp64(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        // Convert ILP64 arguments to LP64
        if (!registered_zgemm_lp64) {
            throw std::runtime_error("zgemm (LP64) symbol not found - BLAS library may not be linked correctly");
        }
        const int m_int = static_cast<int>(*m), n_int = static_cast<int>(*n), k_int = static_cast<int>(*k);
        const int lda_int = static_cast<int>(*lda), ldb_int = static_cast<int>(*ldb), ldc_int = static_cast<int>(*ldc);
        registered_zgemm_lp64(transa, transb, &m_int, &n_int, &k_int, alpha, a, &lda_int, b, &ldb_int, beta, c, &ldc_int);
    }
}

} // namespace sparseir
