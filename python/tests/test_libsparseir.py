import pytest
import numpy as np
import ctypes
from libsparseir import (
    lib,
    Kernel,
    FermionicFiniteTempBasis,
    BosonicFiniteTempBasis,
    spir_order_type
)

def test_kernel_creation():
    # Test logistic kernel creation
    kernel = Kernel.logistic(lambda_=1.0)
    assert kernel is not None

    # Test regularized bose kernel creation
    kernel = Kernel.regularized_bose(lambda_=1.0)
    assert kernel is not None

def test_kernel_domain():
    # Test domain for logistic kernel
    kernel = Kernel.logistic(lambda_=1.0)
    xmin, xmax, ymin, ymax = kernel.domain()
    assert isinstance(xmin, float)
    assert isinstance(xmax, float)
    assert isinstance(ymin, float)
    assert isinstance(ymax, float)
    assert xmin < xmax
    assert ymin < ymax

def test_finite_temp_basis_creation():
    # Test fermionic basis creation
    basis = FermionicFiniteTempBasis.new(beta=10.0, omega_max=1.0, epsilon=1e-6)
    assert basis is not None
    assert basis.size() > 0

    # Test bosonic basis creation
    basis = BosonicFiniteTempBasis.new(beta=10.0, omega_max=1.0, epsilon=1e-6)
    assert basis is not None
    assert basis.size() > 0

def test_sampling_creation(basis):
    # Test tau sampling
    sampling = basis.create_tau_sampling()
    assert sampling is not None

    # Test Matsubara sampling
    sampling = basis.create_matsubara_sampling()
    assert sampling is not None

def test_dlr_creation(basis):
    dlr = basis.create_dlr()
    assert dlr is not None

def test_dlr_conversion(dlr, test_data):
    # Test to_IR conversion
    ir_data = dlr.to_IR(test_data['real'])
    assert ir_data.shape == test_data['real'].shape
    assert ir_data.dtype == np.float64

    # Test from_IR conversion
    output_data = dlr.from_IR(ir_data)
    assert output_data.shape == ir_data.shape
    assert output_data.dtype == np.float64

def test_order_type(sampling, test_data):
    # Test row-major order
    output_row = sampling.evaluate_dd(test_data['real'], order=spir_order_type.SPIR_ORDER_ROW_MAJOR)
    output_col = sampling.evaluate_dd(test_data['real'], order=spir_order_type.SPIR_ORDER_COLUMN_MAJOR)

    assert output_row.shape == output_col.shape
    assert not np.array_equal(output_row, output_col)  # Different orders should give different results

def test_error_handling():
    # Test invalid kernel creation
    with pytest.raises(RuntimeError):
        Kernel.logistic(lambda_=-1.0)  # Invalid lambda

    # Test invalid basis creation
    with pytest.raises(RuntimeError):
        FermionicFiniteTempBasis.new(beta=-1.0, omega_max=1.0, epsilon=1e-6)  # Invalid beta

    # Test invalid sampling evaluation
    basis = FermionicFiniteTempBasis.new(beta=10.0, omega_max=1.0, epsilon=1e-6)
    sampling = basis.create_tau_sampling()
    with pytest.raises(RuntimeError):
        sampling.evaluate_dd(np.array([]))  # Empty input

def test_memory_management():
    # Test that objects are properly cleaned up
    basis = FermionicFiniteTempBasis.new(beta=10.0, omega_max=1.0, epsilon=1e-6)
    sampling = basis.create_tau_sampling()
    dlr = basis.create_dlr()

    # Objects should be cleaned up when they go out of scope
    del basis
    del sampling
    del dlr

def test_sampling_points(sampling):
    # Test getting number of sampling points
    num_points = ctypes.c_int()
    ret = lib.spir_sampling_get_num_points(sampling._ptr, ctypes.byref(num_points))
    assert ret == 0
    assert num_points.value > 0


if __name__ == "__main__":
    pytest.main([__file__])