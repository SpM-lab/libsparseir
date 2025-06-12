"""
Core functionality for the SparseIR Python bindings.
"""

import os
import sys
import ctypes
from ctypes import *
import numpy as np

from .types import *
from .constants import *

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

# Set up function prototypes
def _setup_prototypes():
    # Version function
    _lib.spir_get_version.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
    _lib.spir_get_version.restype = c_int

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
    _lib.spir_sve_result_new.argtypes = [spir_kernel, c_double, POINTER(c_int)]
    _lib.spir_sve_result_new.restype = spir_sve_result

    _lib.spir_sve_result_get_size.argtypes = [spir_sve_result, POINTER(c_int)]
    _lib.spir_sve_result_get_size.restype = c_int

    _lib.spir_sve_result_get_svals.argtypes = [spir_sve_result, POINTER(c_double)]
    _lib.spir_sve_result_get_svals.restype = c_int

    # Basis functions
    _lib.spir_basis_new.argtypes = [
        c_int, c_double, c_double, spir_kernel, spir_sve_result, POINTER(c_int)
    ]
    _lib.spir_basis_new.restype = spir_basis

    _lib.spir_basis_get_size.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_size.restype = c_int

    _lib.spir_basis_get_svals.argtypes = [spir_basis, POINTER(c_double)]
    _lib.spir_basis_get_svals.restype = c_int

    _lib.spir_basis_get_stats.argtypes = [spir_basis, POINTER(c_int)]
    _lib.spir_basis_get_stats.restype = c_int

_setup_prototypes()

def get_version():
    """Get the version information of the SparseIR library.
    
    Returns:
        tuple: A tuple of (major, minor, patch) version numbers.
    """
    major = c_int()
    minor = c_int()
    patch = c_int()
    
    status = _lib.spir_get_version(byref(major), byref(minor), byref(patch))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get version information: {status}")
    
    return major.value, minor.value, patch.value

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

def kernel_domain(kernel):
    """Get the domain boundaries of a kernel."""
    xmin = c_double()
    xmax = c_double()
    ymin = c_double()
    ymax = c_double()
    
    status = _lib.spir_kernel_domain(
        kernel, byref(xmin), byref(xmax), byref(ymin), byref(ymax)
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get kernel domain: {status}")
    
    return xmin.value, xmax.value, ymin.value, ymax.value

def sve_result_new(kernel, epsilon):
    """Create a new SVE result."""
    status = c_int()
    sve = _lib.spir_sve_result_new(kernel, epsilon, byref(status))
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
    svals = np.zeros(size, dtype=DOUBLE_DTYPE)
    status = _lib.spir_sve_result_get_svals(sve, svals.ctypes.data_as(POINTER(c_double)))
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get singular values: {status}")
    return svals

def basis_new(statistics, beta, omega_max, kernel, sve):
    """Create a new basis."""
    status = c_int()
    basis = _lib.spir_basis_new(
        statistics, beta, omega_max, kernel, sve, byref(status)
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
    svals = np.zeros(size, dtype=DOUBLE_DTYPE)
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