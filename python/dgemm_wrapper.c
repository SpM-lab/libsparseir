#include <dlfcn.h>
#include <stdio.h>

// Function pointers for SciPy BLAS functions
static void (*scipy_dgemm_64_ptr)(char *, char *, int *, int *, int *, 
                                  double *, double *, int *, double *, int *, 
                                  double *, double *, int *) = NULL;
static void (*scipy_zgemm_64_ptr)(char *, char *, int *, int *, int *, 
                                  double *, double *, int *, double *, int *, 
                                  double *, double *, int *) = NULL;

// Initialize function pointers
static void init_functions() {
    if (scipy_dgemm_64_ptr == NULL || scipy_zgemm_64_ptr == NULL) {
        // Try to find the symbols in already loaded libraries
        scipy_dgemm_64_ptr = (void (*)(char *, char *, int *, int *, int *, 
                                       double *, double *, int *, double *, int *, 
                                       double *, double *, int *))dlsym(RTLD_DEFAULT, "scipy_dgemm_64_");
        scipy_zgemm_64_ptr = (void (*)(char *, char *, int *, int *, int *, 
                                       double *, double *, int *, double *, int *, 
                                       double *, double *, int *))dlsym(RTLD_DEFAULT, "scipy_zgemm_64_");
        
        if (scipy_dgemm_64_ptr == NULL || scipy_zgemm_64_ptr == NULL) {
            fprintf(stderr, "Warning: Could not find SciPy BLAS functions\n");
        }
    }
}

// Alias dgemm_ to scipy_dgemm_64_
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, 
            double *alpha, double *a, int *lda, double *b, int *ldb, 
            double *beta, double *c, int *ldc) {
    init_functions();
    if (scipy_dgemm_64_ptr != NULL) {
        scipy_dgemm_64_ptr(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

// Alias zgemm_ to scipy_zgemm_64_
void zgemm_(char *transa, char *transb, int *m, int *n, int *k, 
            double *alpha, double *a, int *lda, double *b, int *ldb, 
            double *beta, double *c, int *ldc) {
    init_functions();
    if (scipy_zgemm_64_ptr != NULL) {
        scipy_zgemm_64_ptr(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}
