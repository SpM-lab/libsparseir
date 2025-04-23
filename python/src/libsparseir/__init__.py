from ._libsparseir_cffi import ffi, lib
import numpy as np

# Constants
STATISTICS_FERMIONIC = lib.SPIR_STATISTICS_FERMIONIC
STATISTICS_BOSONIC = lib.SPIR_STATISTICS_BOSONIC
ORDER_COLUMN_MAJOR = lib.SPIR_ORDER_COLUMN_MAJOR
ORDER_ROW_MAJOR = lib.SPIR_ORDER_ROW_MAJOR

class Kernel:
    def __init__(self, ptr):
        self._ptr = ptr

    def __del__(self):
        if self._ptr:
            lib.spir_destroy_kernel(self._ptr)

    @classmethod
    def logistic(cls, lambda_):
        ptr = lib.spir_logistic_kernel_new(lambda_)
        if not ptr:
            raise RuntimeError("Failed to create logistic kernel")
        return cls(ptr)

    @classmethod
    def regularized_bose(cls, lambda_):
        ptr = lib.spir_regularized_bose_kernel_new(lambda_)
        if not ptr:
            raise RuntimeError("Failed to create regularized bose kernel")
        return cls(ptr)

    def domain(self):
        xmin = ffi.new("double*")
        xmax = ffi.new("double*")
        ymin = ffi.new("double*")
        ymax = ffi.new("double*")
        if lib.spir_kernel_domain(self._ptr, xmin, xmax, ymin, ymax) != 0:
            raise RuntimeError("Failed to get kernel domain")
        return (xmin[0], xmax[0], ymin[0], ymax[0])

    def matrix(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        nx = len(x)
        ny = len(y)
        out = np.zeros((nx, ny), dtype=np.float64)

        x_ptr = ffi.cast("double*", x.ctypes.data)
        y_ptr = ffi.cast("double*", y.ctypes.data)
        out_ptr = ffi.cast("double*", out.ctypes.data)

        if lib.spir_kernel_matrix(self._ptr, x_ptr, nx, y_ptr, ny, out_ptr) != 0:
            raise RuntimeError("Failed to compute kernel matrix")
        return out

class FermionicFiniteTempBasis:
    def __init__(self, ptr):
        self._ptr = ptr

    def __del__(self):
        if self._ptr:
            lib.spir_destroy_fermionic_finite_temp_basis(self._ptr)

    @classmethod
    def new(cls, beta, omega_max, epsilon):
        ptr = lib.spir_fermionic_finite_temp_basis_new(beta, omega_max, epsilon)
        if not ptr:
            raise RuntimeError("Failed to create fermionic finite temp basis")
        return cls(ptr)

class BosonicFiniteTempBasis:
    def __init__(self, ptr):
        self._ptr = ptr

    def __del__(self):
        if self._ptr:
            lib.spir_destroy_bosonic_finite_temp_basis(self._ptr)

    @classmethod
    def new(cls, beta, omega_max, epsilon):
        ptr = lib.spir_bosonic_finite_temp_basis_new(beta, omega_max, epsilon)
        if not ptr:
            raise RuntimeError("Failed to create bosonic finite temp basis")
        return cls(ptr)

__all__ = [
    'Kernel',
    'FermionicFiniteTempBasis',
    'BosonicFiniteTempBasis',
    'STATISTICS_FERMIONIC',
    'STATISTICS_BOSONIC',
    'ORDER_COLUMN_MAJOR',
    'ORDER_ROW_MAJOR',
]
