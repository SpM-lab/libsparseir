"""
SparseIR Python bindings

This package provides Python bindings for the SparseIR C library.
"""

from .core import *
from .constants import *
from .ctypes_wrapper import *

# Get version from the library
try:
    __version__ = ".".join(map(str, get_version()))
except Exception:
    # Fallback version if library is not available
    __version__ = "0.0.1" 