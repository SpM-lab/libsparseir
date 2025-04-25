import ctypes
import os
import sys
from typing import Optional, Union, List, Tuple
import numpy as np

# Load the shared library
if sys.platform == "darwin":
    libname = "libsparseir.dylib"
elif sys.platform == "linux":
    libname = "libsparseir.so"
elif sys.platform == "win32":
    libname = "sparseir.dll"
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

# Try to load the library from the build directory
try:
    repository_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )
    libpath = os.path.join(repository_root, "build", libname)
    lib = ctypes.CDLL(libpath)
except OSError as e:
    raise RuntimeError(f"Could not load {libname}: {e}")

# Define ctypes types
c_double = ctypes.c_double
c_int = ctypes.c_int
c_int32 = ctypes.c_int32
c_void_p = ctypes.c_void_p


# Define complex type
class c_complex(ctypes.Structure):
    _fields_ = [("real", c_double), ("imag", c_double)]


# Define enums
class spir_order_type(ctypes.c_int):
    SPIR_ORDER_ROW_MAJOR = 0
    SPIR_ORDER_COLUMN_MAJOR = 1


# Define opaque types
class spir_kernel(ctypes.Structure):
    pass


class spir_sampling(ctypes.Structure):
    pass


class spir_dlr(ctypes.Structure):
    pass


class spir_finite_temp_basis(ctypes.Structure):
    pass


class spir_continuous_functions(ctypes.Structure):
    pass


class spir_matsubara_functions(ctypes.Structure):
    pass


# Define function prototypes
lib.spir_logistic_kernel_new.argtypes = [c_double]
lib.spir_logistic_kernel_new.restype = ctypes.POINTER(spir_kernel)

lib.spir_regularized_bose_kernel_new.argtypes = [c_double]
lib.spir_regularized_bose_kernel_new.restype = ctypes.POINTER(spir_kernel)

lib.spir_kernel_domain.argtypes = [
    ctypes.POINTER(spir_kernel),
    ctypes.POINTER(c_double),
    ctypes.POINTER(c_double),
    ctypes.POINTER(c_double),
    ctypes.POINTER(c_double),
]
lib.spir_kernel_domain.restype = c_int

# Define function prototypes for sampling
lib.spir_fermionic_tau_sampling_new.argtypes = [ctypes.POINTER(spir_finite_temp_basis)]
lib.spir_fermionic_tau_sampling_new.restype = ctypes.POINTER(spir_sampling)

lib.spir_fermionic_matsubara_sampling_new.argtypes = [
    ctypes.POINTER(spir_finite_temp_basis)
]
lib.spir_fermionic_matsubara_sampling_new.restype = ctypes.POINTER(spir_sampling)

lib.spir_sampling_evaluate_dd.argtypes = [
    ctypes.POINTER(spir_sampling),
    spir_order_type,
    c_int32,
    ctypes.POINTER(c_int32),
    c_int32,
    ctypes.POINTER(c_double),
    ctypes.POINTER(c_double),
]
lib.spir_sampling_evaluate_dd.restype = c_int

lib.spir_sampling_evaluate_dz.argtypes = [
    ctypes.POINTER(spir_sampling),
    spir_order_type,
    c_int32,
    ctypes.POINTER(c_int32),
    c_int32,
    ctypes.POINTER(c_double),
    ctypes.POINTER(c_complex),
]
lib.spir_sampling_evaluate_dz.restype = c_int

lib.spir_sampling_evaluate_zz.argtypes = [
    ctypes.POINTER(spir_sampling),
    spir_order_type,
    c_int32,
    ctypes.POINTER(c_int32),
    c_int32,
    ctypes.POINTER(c_complex),
    ctypes.POINTER(c_complex),
]
lib.spir_sampling_evaluate_zz.restype = c_int

# Define function prototypes for DLR
lib.spir_fermionic_dlr_new.argtypes = [ctypes.POINTER(spir_finite_temp_basis)]
lib.spir_fermionic_dlr_new.restype = ctypes.POINTER(spir_dlr)

lib.spir_fermionic_dlr_to_IR.argtypes = [
    ctypes.POINTER(spir_dlr),
    spir_order_type,
    c_int32,
    ctypes.POINTER(c_int32),
    ctypes.POINTER(c_double),
    ctypes.POINTER(c_double),
]
lib.spir_fermionic_dlr_to_IR.restype = c_int

lib.spir_fermionic_dlr_from_IR.argtypes = [
    ctypes.POINTER(spir_dlr),
    spir_order_type,
    c_int32,
    ctypes.POINTER(c_int32),
    ctypes.POINTER(c_double),
    ctypes.POINTER(c_double),
]
lib.spir_fermionic_dlr_from_IR.restype = c_int

# Define function prototypes for finite temperature basis
lib.spir_fermionic_finite_temp_basis_new.argtypes = [c_double, c_double, c_double]
lib.spir_fermionic_finite_temp_basis_new.restype = ctypes.POINTER(
    spir_finite_temp_basis
)

lib.spir_bosonic_finite_temp_basis_new.argtypes = [c_double, c_double, c_double]
lib.spir_bosonic_finite_temp_basis_new.restype = ctypes.POINTER(spir_finite_temp_basis)

lib.spir_fermionic_finite_temp_basis_get_size.argtypes = [
    ctypes.POINTER(spir_finite_temp_basis),
    ctypes.POINTER(c_int),
]
lib.spir_fermionic_finite_temp_basis_get_size.restype = c_int

lib.spir_bosonic_finite_temp_basis_get_size.argtypes = [
    ctypes.POINTER(spir_finite_temp_basis),
    ctypes.POINTER(c_int),
]
lib.spir_bosonic_finite_temp_basis_get_size.restype = c_int


# Define Python classes
class Kernel:
    def __init__(self, ptr: ctypes.POINTER(spir_kernel)):
        self._ptr = ptr

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            lib.spir_destroy_kernel(self._ptr)

    @classmethod
    def logistic(cls, lambda_: float) -> "Kernel":
        ptr = lib.spir_logistic_kernel_new(lambda_)
        if not ptr:
            raise RuntimeError("Failed to create logistic kernel")
        return cls(ptr)

    @classmethod
    def regularized_bose(cls, lambda_: float) -> "Kernel":
        ptr = lib.spir_regularized_bose_kernel_new(lambda_)
        if not ptr:
            raise RuntimeError("Failed to create regularized bose kernel")
        return cls(ptr)

    def domain(self) -> Tuple[float, float, float, float]:
        xmin = c_double()
        xmax = c_double()
        ymin = c_double()
        ymax = c_double()

        ret = lib.spir_kernel_domain(
            self._ptr,
            ctypes.byref(xmin),
            ctypes.byref(xmax),
            ctypes.byref(ymin),
            ctypes.byref(ymax),
        )

        if ret != 0:
            raise RuntimeError("Failed to get kernel domain")

        return (xmin.value, xmax.value, ymin.value, ymax.value)


class Sampling:
    def __init__(self, ptr: ctypes.POINTER(spir_sampling)):
        self._ptr = ptr

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            lib.spir_destroy_sampling(self._ptr)

    def evaluate_dd(
        self,
        input_data: np.ndarray,
        order: spir_order_type = spir_order_type.SPIR_ORDER_ROW_MAJOR,
    ) -> np.ndarray:
        """Evaluate sampling with double input and double output."""
        if not input_data.flags["C_CONTIGUOUS"]:
            input_data = np.ascontiguousarray(input_data)

        input_dims = np.array(input_data.shape, dtype=np.int32)
        ndim = len(input_dims)
        target_dim = 0  # First dimension is the target

        output_shape = list(input_data.shape)
        output = np.empty(output_shape, dtype=np.float64)

        ret = lib.spir_sampling_evaluate_dd(
            self._ptr,
            order,
            ndim,
            input_dims.ctypes.data_as(ctypes.POINTER(c_int32)),
            target_dim,
            input_data.ctypes.data_as(ctypes.POINTER(c_double)),
            output.ctypes.data_as(ctypes.POINTER(c_double)),
        )

        if ret != 0:
            raise RuntimeError("Failed to evaluate sampling")

        return output

    def evaluate_dz(
        self,
        input_data: np.ndarray,
        order: spir_order_type = spir_order_type.SPIR_ORDER_ROW_MAJOR,
    ) -> np.ndarray:
        """Evaluate sampling with double input and complex output."""
        if not input_data.flags["C_CONTIGUOUS"]:
            input_data = np.ascontiguousarray(input_data)

        input_dims = np.array(input_data.shape, dtype=np.int32)
        ndim = len(input_dims)
        target_dim = 0

        output_shape = list(input_data.shape)
        output = np.empty(output_shape, dtype=np.complex128)

        ret = lib.spir_sampling_evaluate_dz(
            self._ptr,
            order,
            ndim,
            input_dims.ctypes.data_as(ctypes.POINTER(c_int32)),
            target_dim,
            input_data.ctypes.data_as(ctypes.POINTER(c_double)),
            output.ctypes.data_as(ctypes.POINTER(c_complex)),
        )

        if ret != 0:
            raise RuntimeError("Failed to evaluate sampling")

        return output

    def evaluate_zz(
        self,
        input_data: np.ndarray,
        order: spir_order_type = spir_order_type.SPIR_ORDER_ROW_MAJOR,
    ) -> np.ndarray:
        """Evaluate sampling with complex input and complex output."""
        if not input_data.flags["C_CONTIGUOUS"]:
            input_data = np.ascontiguousarray(input_data)

        input_dims = np.array(input_data.shape, dtype=np.int32)
        ndim = len(input_dims)
        target_dim = 0

        output_shape = list(input_data.shape)
        output = np.empty(output_shape, dtype=np.complex128)

        ret = lib.spir_sampling_evaluate_zz(
            self._ptr,
            order,
            ndim,
            input_dims.ctypes.data_as(ctypes.POINTER(c_int32)),
            target_dim,
            input_data.ctypes.data_as(ctypes.POINTER(c_complex)),
            output.ctypes.data_as(ctypes.POINTER(c_complex)),
        )

        if ret != 0:
            raise RuntimeError("Failed to evaluate sampling")

        return output


class DLR:
    def __init__(self, ptr: ctypes.POINTER(spir_dlr)):
        self._ptr = ptr

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            pass
            # lib.spir_destroy_dlr(self._ptr)

    def to_IR(
        self,
        input_data: np.ndarray,
        order: spir_order_type = spir_order_type.SPIR_ORDER_ROW_MAJOR,
    ) -> np.ndarray:
        """Convert to IR representation."""
        if not input_data.flags["C_CONTIGUOUS"]:
            input_data = np.ascontiguousarray(input_data)

        input_dims = np.array(input_data.shape, dtype=np.int32)
        ndim = len(input_dims)

        output_shape = list(input_data.shape)
        output = np.empty(output_shape, dtype=np.float64)

        ret = lib.spir_fermionic_dlr_to_IR(
            self._ptr,
            order,
            ndim,
            input_dims.ctypes.data_as(ctypes.POINTER(c_int32)),
            input_data.ctypes.data_as(ctypes.POINTER(c_double)),
            output.ctypes.data_as(ctypes.POINTER(c_double)),
        )

        if ret != 0:
            raise RuntimeError("Failed to convert to IR")

        return output

    def from_IR(
        self,
        input_data: np.ndarray,
        order: spir_order_type = spir_order_type.SPIR_ORDER_ROW_MAJOR,
    ) -> np.ndarray:
        """Convert from IR representation."""
        if not input_data.flags["C_CONTIGUOUS"]:
            input_data = np.ascontiguousarray(input_data)

        input_dims = np.array(input_data.shape, dtype=np.int32)
        ndim = len(input_dims)

        output_shape = list(input_data.shape)
        output = np.empty(output_shape, dtype=np.float64)

        ret = lib.spir_fermionic_dlr_from_IR(
            self._ptr,
            order,
            ndim,
            input_dims.ctypes.data_as(ctypes.POINTER(c_int32)),
            input_data.ctypes.data_as(ctypes.POINTER(c_double)),
            output.ctypes.data_as(ctypes.POINTER(c_double)),
        )

        if ret != 0:
            raise RuntimeError("Failed to convert from IR")

        return output


class FiniteTempBasis:
    def __init__(self, ptr: ctypes.POINTER(spir_finite_temp_basis)):
        self._ptr = ptr

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            pass
            # lib.spir_destroy_finite_temp_basis(self._ptr)

    def size(self) -> int:
        """Get the size of the basis."""
        size = c_int()
        ret = lib.spir_fermionic_finite_temp_basis_get_size(
            self._ptr, ctypes.byref(size)
        )
        if ret != 0:
            raise RuntimeError("Failed to get basis size")
        return size.value

    def create_tau_sampling(self) -> "Sampling":
        """Create a tau sampling object."""
        ptr = lib.spir_fermionic_tau_sampling_new(self._ptr)
        if not ptr:
            raise RuntimeError("Failed to create tau sampling")
        return Sampling(ptr)

    def create_matsubara_sampling(self) -> "Sampling":
        """Create a Matsubara sampling object."""
        ptr = lib.spir_fermionic_matsubara_sampling_new(self._ptr)
        if not ptr:
            raise RuntimeError("Failed to create Matsubara sampling")
        return Sampling(ptr)

    def create_dlr(self) -> "DLR":
        """Create a DLR object."""
        ptr = lib.spir_fermionic_dlr_new(self._ptr)
        if not ptr:
            raise RuntimeError("Failed to create DLR")
        return DLR(ptr)


class FermionicFiniteTempBasis(FiniteTempBasis):
    @classmethod
    def new(
        cls, beta: float, omega_max: float, epsilon: float
    ) -> "FermionicFiniteTempBasis":
        """Create a new fermionic finite temperature basis."""
        ptr = lib.spir_fermionic_finite_temp_basis_new(beta, omega_max, epsilon)
        if not ptr:
            raise RuntimeError("Failed to create fermionic finite temperature basis")
        return cls(ptr)


class BosonicFiniteTempBasis(FiniteTempBasis):
    @classmethod
    def new(
        cls, beta: float, omega_max: float, epsilon: float
    ) -> "BosonicFiniteTempBasis":
        """Create a new bosonic finite temperature basis."""
        ptr = lib.spir_bosonic_finite_temp_basis_new(beta, omega_max, epsilon)
        if not ptr:
            raise RuntimeError("Failed to create bosonic finite temperature basis")
        return cls(ptr)


# Export the main classes
__all__ = [
    "Kernel",
    "Sampling",
    "DLR",
    "FiniteTempBasis",
    "FermionicFiniteTempBasis",
    "BosonicFiniteTempBasis",
    "spir_order_type",
]
