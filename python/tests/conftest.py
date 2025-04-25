import pytest
import numpy as np
import ctypes
import sys
import os
from libsparseir import (
    FermionicFiniteTempBasis,
    Kernel,
    spir_order_type
)

# Define ctypes types
c_double = ctypes.c_double
c_int = ctypes.c_int
c_int32 = ctypes.c_int32


@pytest.fixture
def basis():
    """Create a fermionic finite temperature basis for testing."""
    return FermionicFiniteTempBasis.new(beta=10.0, omega_max=1.0, epsilon=1e-6)

@pytest.fixture
def sampling(basis):
    """Create a tau sampling object for testing."""
    return basis.create_tau_sampling()

@pytest.fixture
def dlr(basis):
    """Create a DLR object for testing."""
    return basis.create_dlr()

@pytest.fixture
def kernel():
    """Create a kernel object for testing."""
    return Kernel.logistic(lambda_=1.0)

@pytest.fixture
def test_data():
    """Create test data arrays."""
    return {
        'real': np.random.rand(10, 10),
        'complex': np.random.rand(10, 10) + 1j * np.random.rand(10, 10),
        'small': np.random.rand(5, 5),
        'large': np.random.rand(20, 20),
        '1d': np.random.rand(10),
        '3d': np.random.rand(5, 5, 5)
    }

@pytest.fixture
def matsubara_points():
    """Create Matsubara frequency points for testing."""
    return np.array([0, 1, 2, 3, 4], dtype=np.int32)

@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    # Setup code here if needed
    yield
    # Teardown code here if needed