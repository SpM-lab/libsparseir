"""
Abstract base class for basis objects.

This module provides the abstract interface that all basis types should implement.
"""

from abc import ABC, abstractmethod

class AbstractKernel(ABC):
    """Abstract base class for kernels."""
    pass

class AbstractBasis(ABC):
    r"""Abstract base class for bases on the imaginary-time axis.

    This class stores a set of basis functions. We can then expand a two-point
    propagator G(τ), where τ is imaginary time:

        G(τ) ≈ Σ_{l=0}^{L-1} g_l U_l(τ)

    where U is now the l-th basis function, stored in u and g denote the
    expansion coefficients. Similarly, the Fourier transform Ĝ(n), where n
    is a reduced Matsubara frequency, can be expanded as follows:

        Ĝ(n) ≈ Σ_{l=0}^{L-1} g_l Û_l(n)

    where Û is the Fourier transform of the l-th basis function, stored
    in uhat.

    Assuming that basis is an instance of some abstract basis, g is a vector
    of expansion coefficients, tau is some imaginary time and n some frequency,
    we can write this in the library as follows:

        G_tau = basis.u(tau).T @ gl
        G_n = basis.uhat(n).T @ gl
    """

    @property
    @abstractmethod
    def u(self):
        r"""Basis functions on the imaginary time axis.

        Set of IR basis functions on the imaginary time (tau) axis, where tau
        is a real number between zero and beta. To get the l-th basis function
        at imaginary time tau of some basis, use:

            ultau = basis.u[l](tau)        # l-th basis function at time tau

        Note that u supports vectorization both over l and tau.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def uhat(self):
        r"""Basis functions on the reduced Matsubara frequency (wn) axis.

        Set of IR basis functions reduced Matsubara frequency (wn) axis, where
        wn is an integer. These are related to u by the following Fourier transform:

           û(n) = ∫₀^β dτ exp(iπnτ/β) u(τ)

        To get the l-th basis function at some reduced frequency wn of
        some basis, use:

            uln = basis.uhat[l](wn)        # l-th basis function at freq wn

        Note:
            Instead of the value of the Matsubara frequency, these functions
            expect integers corresponding to the prefactor of pi over beta.
            For example, the first few positive fermionic frequencies would
            be specified as [1, 3, 5, 7], and the first bosonic frequencies
            are [0, 2, 4, 6]. This is also distinct to an index!
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def statistics(self):
        """Quantum statistic ("F" for fermionic, "B" for bosonic)"""
        raise NotImplementedError()

    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.

        This can be used to truncate the basis to the n most significant
        singular values: basis[:3].
        """
        raise NotImplementedError()

    @property
    def shape(self):
        """Shape of the basis function set"""
        return (self.size,)

    @property
    @abstractmethod
    def size(self):
        """Number of basis functions / singular values"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def significance(self):
        """Significances of the basis functions

        Vector of significance values, one for each basis function. Each
        value is a number between 0 and 1 which is an a-priori bound on the
        (relative) error made by discarding the associated coefficient.
        """
        raise NotImplementedError()

    @property
    def accuracy(self):
        """Accuracy of the basis.

        Upper bound to the relative error of representing a propagator with
        the given number of basis functions (number between 0 and 1).
        """
        return self.significance[-1]

    @property
    @abstractmethod
    def lambda_(self):
        """Basis cutoff parameter, Λ = β * wmax, or None if not present"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def beta(self):
        """Inverse temperature"""
        raise NotImplementedError()

    @property
    def wmax(self):
        """Real frequency cutoff or None if not present"""
        if self.lambda_ is None or self.beta is None:
            return None
        return self.lambda_ / self.beta

    @abstractmethod
    def default_tau_sampling_points(self, *, npoints=None):
        """Default sampling points on the imaginary time axis

        Parameters
        ----------
        npoints : int, optional
            Minimum number of sampling points to return.
        """
        raise NotImplementedError()

    @abstractmethod
    def default_matsubara_sampling_points(self, *, npoints=None,
                                          positive_only=False):
        """Default sampling points on the imaginary frequency axis

        Parameters
        ----------
        npoints : int, optional
            Minimum number of sampling points to return.
        positive_only : bool
            Only return non-negative frequencies. This is useful if the
            object to be fitted is symmetric in Matsubara frequency,
            ghat(w) == ghat(-w).conj(), or, equivalently, real in
            imaginary time.
        """
        raise NotImplementedError()

    @property
    def is_well_conditioned(self):
        """Returns True if the sampling is expected to be well-conditioned"""
        return True