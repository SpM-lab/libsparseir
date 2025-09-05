"""
SparseIR Python bindings

This package provides Python bindings for the SparseIR C library.
"""

from .abstract import AbstractBasis
from .basis import FiniteTempBasis, finite_temp_bases
from .sampling import TauSampling, MatsubaraSampling
from .kernel import LogisticKernel, RegularizedBoseKernel
from .sve import SVEResult, compute_sve
from .basis_set import FiniteTempBasisSet

# New augmented functionality
from .augment import (
    AugmentedBasis, AugmentedTauFunction, AugmentedMatsubaraFunction,
    AbstractAugmentation, TauConst, TauLinear, MatsubaraConst
)

# DLR functionality
from .dlr import (
    DiscreteLehmannRepresentation
)

# Export list for better documentation
__all__ = [
    # Core functionality
    'AbstractBasis', 'FiniteTempBasis', 'finite_temp_bases',
    'TauSampling', 'MatsubaraSampling', 'FiniteTempBasisSet',
    'LogisticKernel', 'RegularizedBoseKernel',
    'SVEResult', 'compute_sve',

    # Augmented functionality
    'AugmentedBasis', 'AugmentedTauFunction', 'AugmentedMatsubaraFunction',
    'AbstractAugmentation', 'TauConst', 'TauLinear', 'MatsubaraConst',

    # DLR functionality
    'DiscreteLehmannRepresentation', 'TauPoles', 'MatsubaraPoles',
]