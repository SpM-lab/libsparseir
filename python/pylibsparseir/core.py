"""
Core functionality for the SparseIR Python bindings.
"""

import os
import sys
import ctypes
from ctypes import c_int, c_double, c_int64, c_size_t, c_bool, POINTER, byref
from ctypes import CDLL
import numpy as np

from .ctypes_wrapper import spir_kernel, spir_sve_result, spir_basis, spir_funcs, spir_sampling
from pylibsparseir.constants import COMPUTATION_SUCCESS, SPIR_ORDER_ROW_MAJOR, SPIR_ORDER_COLUMN_MAJOR, SPIR_TWORK_FLOAT64, SPIR_TWORK_FLOAT64X2, SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC

def _find_library():
    """Find the SparseIR shared library."""
    if sys.platform == "darwin":
        libname = "libsparseir.dylib"
    elif sys.platform == "win32":
        libname = "sparseir.dll"
    else:
        libname = "libsparseir.so"

    # Try to find the library in common locations
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build"),
    ]

    for path in search_paths:
        libpath = os.path.join(path, libname)
        if os.path.exists(libpath):
            return libpath

    raise RuntimeError(f"Could not find {libname} in {search_paths}")

# Load the library
try:
    _lib = CDLL(_find_library())
except Exception as e:
    raise RuntimeError(f"Failed to load SparseIR library: {e}")

class c_double_complex(ctypes.Structure):
    """complex is a c structure
    https://docs.python.org/3/library/ctypes.html#module-ctypes suggests
    to use ctypes.Structure to pass structures (and, therefore, complex)
    See: https://stackoverflow.com/questions/13373291/complex-number-in-ctypes
    """
    _fields_ = [("real", ctypes.c_double),("imag", ctypes.c_double)]
    @property
    def value(self):
        return self.real+1j*self.imag # fields declared above

# Set up function prototypes
def _setup_prototypes():
    # Kernel functions
    _lib.spir_logistic_kernel_new.argtypes = [c_double, POINTER(c_int)]
    _lib.spir_logistic_kernel_new.restype = spir_kernel

    _lib.spir_reg_bose_kernel_new.argtypes = [c_double, POINTER(c_int)]
    _lib.spir_reg_bose_kernel_new.restype = spir_kernel

    _lib.spir_kernel_domain.argtypes = [
        spir_kernel, POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_kernel_domain.restype = c_int

    # SVE result functions
    _lib.spir_sve_result_new.argtypes = [spir_kernel, c_double, c_double, c_int, c_int, c_int, POINTER(c_int)]
    _lib.spir_sve_result_new.restype = spir_sve_result

    _lib.spir_sve_result_get_size.argtypes = [spir_sve_result, POINTER(c_int)]
    _lib.spir_sve_result_get_size.restype = c_int

    _lib.spir_sve_result_get_svals.argtypes = [spir_sve_result, POINTER(c_double)]
    _lib.spir_sve_result_get_svals.restype = c_int

    # Basis functions
    _lib.spir_basis_new.argtypes = [
        c_int, c_double, c_double, spir_kernel, spir_sve_result, c_int, POINTER(c_int)
    ]
    _lib.spir_basis_new.restype = spir_basis

    _lib.spir_basis_get_size.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_size.restype = c_int

    _lib.spir_basis_get_svals.argtypes = [spir_basis, POINTER(c_double)]
    _lib.spir_basis_get_svals.restype = c_int

    _lib.spir_basis_get_stats.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_stats.restype = c_int

    # Basis function objects
    _lib.spir_basis_get_u.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_u.restype = spir_funcs

    _lib.spir_basis_get_v.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_v.restype = spir_funcs

    _lib.spir_basis_get_uhat.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_uhat.restype = spir_funcs

    _lib.spir_funcs_get_slice.argtypes = [spir_funcs, c_int, POINTER(c_int), POINTER(c_int)]
    _lib.spir_funcs_get_slice.restype = spir_funcs

    # Function evaluation
    _lib.spir_funcs_get_size.argtypes = [spir_funcs, POINTER(c_int)]
    _lib.spir_funcs_get_size.restype = c_int

    _lib.spir_funcs_eval.argtypes = [spir_funcs, c_double, POINTER(c_double)]
    _lib.spir_funcs_eval.restype = c_int

    _lib.spir_funcs_eval_matsu.argtypes = [spir_funcs, c_int64, POINTER(c_double_complex)]
    _lib.spir_funcs_eval_matsu.restype = c_int

    _lib.spir_funcs_batch_eval.argtypes = [
        spir_funcs, c_int, c_size_t, POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_funcs_batch_eval.restype = c_int

    _lib.spir_funcs_batch_eval_matsu.argtypes = [
        spir_funcs, c_int, c_int, POINTER(c_int64), POINTER(c_double)
    ]
    _lib.spir_funcs_batch_eval_matsu.restype = c_int

    _lib.spir_funcs_get_n_roots.argtypes = [spir_funcs, POINTER(c_int)]
    _lib.spir_funcs_get_n_roots.restype = c_int

    _lib.spir_funcs_get_roots.argtypes = [spir_funcs, POINTER(c_double)]
    _lib.spir_funcs_get_roots.restype = c_int

    # Default sampling points
    _lib.spir_basis_get_n_default_taus.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_n_default_taus.restype = c_int

    _lib.spir_basis_get_default_taus.argtypes = [spir_basis, POINTER(c_double)]
    _lib.spir_basis_get_default_taus.restype = c_int

    _lib.spir_basis_get_default_taus_ext.argtypes = [spir_basis, c_int, POINTER(c_double), POINTER(c_int)]
    _lib.spir_basis_get_default_taus_ext.restype = c_int

    _lib.spir_basis_get_n_default_ws.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_n_default_ws.restype = c_int

    _lib.spir_basis_get_default_ws.argtypes = [spir_basis, POINTER(c_double)]
    _lib.spir_basis_get_default_ws.restype = c_int

    _lib.spir_basis_get_n_default_matsus.argtypes = [spir_basis, c_bool, POINTER(c_int)]
    _lib.spir_basis_get_n_default_matsus.restype = c_int

    _lib.spir_basis_get_n_default_matsus_ext.argtypes = [spir_basis, c_bool, c_int, POINTER(c_int)]
    _lib.spir_basis_get_n_default_matsus_ext.restype = c_int

    _lib.spir_basis_get_default_matsus.argtypes = [spir_basis, c_bool, POINTER(c_int64)]
    _lib.spir_basis_get_default_matsus.restype = c_int

    _lib.spir_basis_get_default_matsus_ext.argtypes = [spir_basis, c_bool, c_int, POINTER(c_int64), POINTER(c_int)]
    _lib.spir_basis_get_default_matsus_ext.restype = c_int

    # Sampling objects
    _lib.spir_tau_sampling_new.argtypes = [spir_basis, c_int, POINTER(c_double), POINTER(c_int)]
    _lib.spir_tau_sampling_new.restype = spir_sampling

    _lib.spir_tau_sampling_new_with_matrix.argtypes = [c_int, c_int, c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_int)]
    _lib.spir_tau_sampling_new_with_matrix.restype = spir_sampling

    _lib.spir_matsu_sampling_new.argtypes = [spir_basis, c_bool, c_int, POINTER(c_int64), POINTER(c_int)]
    _lib.spir_matsu_sampling_new.restype = spir_sampling

    _lib.spir_matsu_sampling_new_with_matrix.argtypes = [
        c_int,                          # order
        c_int,                          # statistics
        c_int,                          # basis_size
        c_bool,                         # positive_only
        c_int,                          # num_points
        POINTER(c_int64),              # points
        POINTER(c_double_complex),     # matrix
        POINTER(c_int)                 # status
    ]
    _lib.spir_matsu_sampling_new_with_matrix.restype = spir_sampling

    # Sampling operations
    _lib.spir_sampling_eval_dd.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_sampling_eval_dd.restype = c_int

    _lib.spir_sampling_fit_dd.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_sampling_fit_dd.restype = c_int

    # Additional sampling functions
    _lib.spir_sampling_get_npoints.argtypes = [spir_sampling, POINTER(c_int)]
    _lib.spir_sampling_get_npoints.restype = c_int

    _lib.spir_sampling_get_taus.argtypes = [spir_sampling, POINTER(c_double)]
    _lib.spir_sampling_get_taus.restype = c_int

    _lib.spir_sampling_get_matsus.argtypes = [spir_sampling, POINTER(c_int64)]
    _lib.spir_sampling_get_matsus.restype = c_int

    _lib.spir_sampling_get_cond_num.argtypes = [spir_sampling, POINTER(c_double)]
    _lib.spir_sampling_get_cond_num.restype = c_int

    # Multi-dimensional sampling evaluation functions
    _lib.spir_sampling_eval_dz.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double_complex)
    ]
    _lib.spir_sampling_eval_dz.restype = c_int

    _lib.spir_sampling_eval_zz.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double_complex), POINTER(c_double_complex)
    ]
    _lib.spir_sampling_eval_zz.restype = c_int

    _lib.spir_sampling_fit_zz.argtypes = [
        spir_sampling, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double_complex), POINTER(c_double_complex)
    ]
    _lib.spir_sampling_fit_zz.restype = c_int

    # DLR functions
    _lib.spir_dlr_new.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_dlr_new.restype = spir_basis

    _lib.spir_dlr_new_with_poles.argtypes = [spir_basis, c_int, POINTER(c_double), POINTER(c_int)]
    _lib.spir_dlr_new_with_poles.restype = spir_basis

    _lib.spir_dlr_get_npoles.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_dlr_get_npoles.restype = c_int

    _lib.spir_dlr_get_poles.argtypes = [spir_basis, POINTER(c_double)]
    _lib.spir_dlr_get_poles.restype = c_int

    _lib.spir_dlr2ir_dd.argtypes = [
        spir_basis, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_dlr2ir_dd.restype = c_int

    _lib.spir_dlr2ir_zz.argtypes = [
        spir_basis, c_int, c_int, POINTER(c_int), c_int,
        POINTER(c_double), POINTER(c_double)
    ]
    _lib.spir_dlr2ir_zz.restype = c_int

    # Release functions
    _lib.spir_kernel_release.argtypes = [spir_kernel]
    _lib.spir_kernel_release.restype = None

    _lib.spir_sve_result_release.argtypes = [spir_sve_result]
    _lib.spir_sve_result_release.restype = None

    _lib.spir_basis_release.argtypes = [spir_basis]
    _lib.spir_basis_release.restype = None

    _lib.spir_funcs_release.argtypes = [spir_funcs]
    _lib.spir_funcs_release.restype = None

    _lib.spir_sampling_release.argtypes = [spir_sampling]
    _lib.spir_sampling_release.restype = None

_setup_prototypes()

# Python wrapper functions
def logistic_kernel_new(lambda_val):
    """Create a new logistic kernel."""
    status = c_int()
    kernel = _lib.spir_logistic_kernel_new(lambda_val, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create logistic kernel: {status.value}")
    return kernel

def reg_bose_kernel_new(lambda_val):
    """Create a new regularized bosonic kernel."""
    status = c_int()
    kernel = _lib.spir_reg_bose_kernel_new(lambda_val, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create regularized bosonic kernel: {status.value}")
    return kernel

def sve_result_new(kernel, epsilon, cutoff=None, lmax=None, n_gauss=None, Twork=None):
    """Create a new SVE result."""
    # Validate epsilon
    if epsilon <= 0:
        raise RuntimeError(f"Failed to create SVE result: epsilon must be positive, got {epsilon}")

    if cutoff is None:
        cutoff = -1.0
    if lmax is None:
        lmax = -1
    if n_gauss is None:
        n_gauss = -1
    if Twork is None:
        Twork = SPIR_TWORK_FLOAT64X2

    status = c_int()
    sve = _lib.spir_sve_result_new(kernel, epsilon, cutoff, lmax, n_gauss, Twork, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create SVE result: {status.value}")
    return sve

def sve_result_get_size(sve):
    """Get the size of an SVE result."""
    size = c_int()
    status = _lib.spir_sve_result_get_size(sve, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get SVE result size: {status}")
    return size.value

def sve_result_get_svals(sve):
    """Get the singular values from an SVE result."""
    size = sve_result_get_size(sve)
    svals = np.zeros(size, dtype=np.float64)
    status = _lib.spir_sve_result_get_svals(sve, svals.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get singular values: {status}")
    return svals

def basis_new(statistics, beta, omega_max, kernel, sve, max_size):
    """Create a new basis."""
    status = c_int()
    basis = _lib.spir_basis_new(
        statistics, beta, omega_max, kernel, sve, max_size, byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create basis: {status.value}")
    return basis

def basis_get_size(basis):
    """Get the size of a basis."""
    size = c_int()
    status = _lib.spir_basis_get_size(basis, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get basis size: {status}")
    return size.value

def basis_get_svals(basis):
    """Get the singular values of a basis."""
    size = basis_get_size(basis)
    svals = np.zeros(size, dtype=np.float64)
    status = _lib.spir_basis_get_svals(basis, svals.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get singular values: {status}")
    return svals


def basis_get_stats(basis):
    """Get the statistics type of a basis."""
    stats = c_int()
    status = _lib.spir_basis_get_stats(basis, byref(stats))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get basis statistics: {status}")
    return stats.value

def basis_get_u(basis):
    """Get the imaginary-time basis functions."""
    status = c_int()
    funcs = _lib.spir_basis_get_u(basis, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get u basis functions: {status.value}")
    return funcs

def basis_get_v(basis):
    """Get the real-frequency basis functions."""
    status = c_int()
    funcs = _lib.spir_basis_get_v(basis, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get v basis functions: {status.value}")
    return funcs

def basis_get_uhat(basis):
    """Get the Matsubara frequency basis functions."""
    status = c_int()
    funcs = _lib.spir_basis_get_uhat(basis, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get uhat basis functions: {status.value}")
    return funcs

def funcs_get_size(funcs):
    """Get the size of a basis function set."""
    size = c_int()
    status = _lib.spir_funcs_get_size(funcs, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get function size: {status}")
    return size.value

# TODO: Rename funcs_eval_single
def funcs_eval_single_float64(funcs, x):
    """Evaluate basis functions at a single point."""
    # Get number of functions
    size = c_int()
    status = _lib.spir_funcs_get_size(funcs, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get function size: {status}")

    # Prepare output array
    out = np.zeros(size.value, dtype=np.float64)

    # Evaluate
    status = _lib.spir_funcs_eval(
        funcs, c_double(x),
        out.ctypes.data_as(POINTER(c_double))
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to evaluate functions: {status}")

    return out

# TODO: Rename to funcs_eval_matsu_single
def funcs_eval_single_complex128(funcs, x):
    """Evaluate basis functions at a single point."""
    # Get number of functions
    size = c_int()
    status = _lib.spir_funcs_get_size(funcs, byref(size))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get function size: {status}")

    # Prepare output array
    out = np.zeros(size.value, dtype=np.complex128)

    # Evaluate
    status = _lib.spir_funcs_eval_matsu(
        funcs, c_int64(x),
        out.ctypes.data_as(POINTER(c_double_complex))
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to evaluate functions: {status}")

    return out

def funcs_get_n_roots(funcs):
    """Get the number of roots of the basis functions."""
    n_roots = c_int()
    status = _lib.spir_funcs_get_n_roots(funcs, byref(n_roots))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get number of roots: {status}")
    return n_roots.value

def funcs_get_roots(funcs):
    """Get the roots of the basis functions."""
    n_roots = funcs_get_n_roots(funcs)
    roots = np.zeros(n_roots, dtype=np.float64)
    status = _lib.spir_funcs_get_roots(funcs, roots.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get roots: {status}")
    return roots

def basis_get_default_tau_sampling_points(basis):
    """Get default tau sampling points for a basis."""
    # Get number of points
    n_points = c_int()
    status = _lib.spir_basis_get_n_default_taus(basis, byref(n_points))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get number of default tau points: {status}")

    # Get the points
    points = np.zeros(n_points.value, dtype=np.float64)
    status = _lib.spir_basis_get_default_taus(basis, points.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default tau points: {status}")

    return points

def basis_get_default_tau_sampling_points_ext(basis, n_points):
    """Get default tau sampling points for a basis."""
    points = np.zeros(n_points, dtype=np.float64)
    n_points_returned = c_int()
    status = _lib.spir_basis_get_default_taus_ext(basis, n_points, points.ctypes.data_as(POINTER(c_double)), byref(n_points_returned))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default tau points: {status}")
    return points

def basis_get_default_omega_sampling_points(basis):
    """Get default omega (real frequency) sampling points for a basis."""
    # Get number of points
    n_points = c_int()
    status = _lib.spir_basis_get_n_default_ws(basis, byref(n_points))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get number of default omega points: {status}")

    # Get the points
    points = np.zeros(n_points.value, dtype=np.float64)
    status = _lib.spir_basis_get_default_ws(basis, points.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default omega points: {status}")

    return points

def basis_get_default_matsubara_sampling_points(basis, positive_only=False):
    """Get default Matsubara sampling points for a basis."""
    # Get number of points
    n_points = c_int()
    status = _lib.spir_basis_get_n_default_matsus(basis, c_bool(positive_only), byref(n_points))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get number of default Matsubara points: {status}")

    # Get the points
    points = np.zeros(n_points.value, dtype=np.int64)
    status = _lib.spir_basis_get_default_matsus(basis, c_bool(positive_only), points.ctypes.data_as(POINTER(c_int64)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default Matsubara points: {status}")

    return points

def basis_get_n_default_matsus_ext(basis, n_points, positive_only):
    """Get the number of default Matsubara sampling points for a basis."""
    n_points_returned = c_int()
    status = _lib.spir_basis_get_n_default_matsus_ext(basis, c_bool(positive_only), n_points, byref(n_points_returned))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get number of default Matsubara points: {status}")
    return n_points_returned.value

def basis_get_default_matsus_ext(basis, positive_only, points):
    n_points = len(points)
    n_points_returned = c_int()
    status = _lib.spir_basis_get_default_matsus_ext(basis, c_bool(positive_only), n_points, points.ctypes.data_as(POINTER(c_int64)), byref(n_points_returned))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get default Matsubara points: {status}")
    return points

def tau_sampling_new(basis, sampling_points=None):
    """Create a new tau sampling object."""
    if sampling_points is None:
        sampling_points = basis_get_default_tau_sampling_points(basis)

    sampling_points = np.asarray(sampling_points, dtype=np.float64)
    n_points = len(sampling_points)

    status = c_int()
    sampling = _lib.spir_tau_sampling_new(
        basis, n_points,
        sampling_points.ctypes.data_as(POINTER(c_double)),
        byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create tau sampling: {status.value}")

    return sampling

def _statistics_to_c(statistics):
    """Convert statistics to c type."""
    if statistics == "F":
        return SPIR_STATISTICS_FERMIONIC
    elif statistics == "B":
        return SPIR_STATISTICS_BOSONIC
    else:
        raise ValueError(f"Invalid statistics: {statistics}")

def tau_sampling_new_with_matrix(basis, statistics, sampling_points, matrix):
    """Create a new tau sampling object with a matrix."""
    status = c_int()
    sampling = _lib.spir_tau_sampling_new_with_matrix(
        SPIR_ORDER_ROW_MAJOR,
        _statistics_to_c(statistics),
        basis.size,
        sampling_points.size,
        sampling_points.ctypes.data_as(POINTER(c_double)),
        matrix.ctypes.data_as(POINTER(c_double)),
        byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create tau sampling: {status.value}")

    return sampling

def matsubara_sampling_new(basis, positive_only=False, sampling_points=None):
    """Create a new Matsubara sampling object."""
    if sampling_points is None:
        sampling_points = basis_get_default_matsubara_sampling_points(basis, positive_only)

    sampling_points = np.asarray(sampling_points, dtype=np.int64)
    n_points = len(sampling_points)

    status = c_int()
    sampling = _lib.spir_matsu_sampling_new(
        basis, c_bool(positive_only), n_points,
        sampling_points.ctypes.data_as(POINTER(c_int64)),
        byref(status)
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create Matsubara sampling: {status.value}")

    return sampling

def matsubara_sampling_new_with_matrix(statistics, basis_size, positive_only, sampling_points, matrix):
    """Create a new Matsubara sampling object with a matrix."""
    status = c_int()
    sampling = _lib.spir_matsu_sampling_new_with_matrix(
        SPIR_ORDER_ROW_MAJOR,                           # order
        _statistics_to_c(statistics),                   # statistics
        c_int(basis_size),                              # basis_size
        c_bool(positive_only),                          # positive_only
        c_int(len(sampling_points)),                    # num_points
        sampling_points.ctypes.data_as(POINTER(c_int64)), # points
        matrix.ctypes.data_as(POINTER(c_double_complex)), # matrix
        byref(status)                                   # status
    )
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to create Matsubara sampling: {status.value}")

    return sampling
