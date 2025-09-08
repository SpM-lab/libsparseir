"""
Configuration and fixtures for pysparseir tests.

This file provides shared fixtures that are available to all tests.
Following the pattern from sparse-ir test suite.
"""

import pytest
import numpy as np
import pylibsparseir


@pytest.fixture(scope="session")
def test_bases():
    """Precomputed test bases for common parameter sets."""
    print("Precomputing test bases ...")
    bases = {}

    test_params = [
        ('F', 1.0, 10.0, 1e-6),    # Small fermion
        ('F', 1.0, 42.0, 1e-8),    # Medium fermion
        ('B', 1.0, 10.0, 1e-6),    # Small boson
        ('F', 4.0, 20.0, 1e-6),    # Different beta
    ]

    for stat, beta, wmax, eps in test_params:
        try:
            basis = pylibsparseir.FiniteTempBasis(stat, beta, wmax, eps)
            bases[(stat, beta, wmax)] = basis
        except Exception as e:
            print(f"Failed to create basis {(stat, beta, wmax)}: {e}")

    return bases


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.RandomState(42)


# Test parameter sets following sparse-ir patterns
KERNEL_LAMBDAS = [10, 42, 1000]
BASIS_PARAMS = [
    ('F', 1.0, 10.0),
    ('F', 1.0, 42.0),
    ('B', 1.0, 10.0),
    ('F', 4.0, 20.0),
]
SAMPLING_PARAMS = [
    ('F', 1.0, 42.0, False),
    ('F', 1.0, 42.0, True),
    ('B', 1.0, 10.0, False),
]