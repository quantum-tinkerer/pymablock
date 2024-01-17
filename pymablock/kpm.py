from typing import Optional, Callable, Union
from collections.abc import Iterator

from scipy import sparse
import numpy as np


def greens_function(
    hamiltonian: Union[np.ndarray, sparse.spmatrix],
    energy: float,
    vector: np.ndarray,
    num_moments: int = 100,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a solution of ``(hamiltonian - energy) @ x = vector``.

    Uses the Kernel polynomial method (KPM) with the Jackson kernel.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian with shape ``(N, N)``. It is required that the Hamiltonian
        is rescaled so that its spectrum lies in the interval ``[-1, 1]``.
    energy :
        Rescaled energy at which to evaluate the Green's function.
    vector :
        Vector with shape ``(N,)``.
    num_moments :
        Number of moments to use in the KPM expansion.

    Returns
    -------
    solution : `~numpy.ndarray`
        Solution of the linear system.
    """
    prefactor = -2 / np.sqrt(1 - energy**2)
    coef = prefactor * np.sin(np.arange(num_moments) * np.arccos(energy))
    coef[0] /= 2
    coef *= jackson_kernel(num_moments)

    return sum(vec * c for c, vec in zip(coef, kpm_vectors(hamiltonian, vector)))


def kpm_vectors(
    hamiltonian: Union[np.ndarray, sparse.spmatrix],
    vector: np.ndarray,
) -> Iterator[np.ndarray]:
    r"""
    Generator of vectors for the Kernel Polynomial Method (KPM).

    This generator yields vectors as :math:`T_n(H) \lvert v \rangle`.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian, dense or sparse array with shape ``(N, N)``.
    vector :
        Vector of length ``N``.

    Yields
    ------
    expanded_vectors : Iterable
        Infinite sequence of Chebyshev polynomials of the Hamiltonian applied
        to the vector.
    """
    yield (alpha_prev := vector)
    yield (alpha := hamiltonian @ alpha_prev)
    while True:
        alpha, alpha_prev = 2 * hamiltonian @ alpha - alpha_prev, alpha
        yield alpha


def rescale(
    hamiltonian: Union[np.ndarray, sparse.spmatrix],
    eps: Optional[float] = 0.01,
    bounds: Optional[tuple[float, float]] = None,
) -> tuple[Union[np.ndarray, sparse.spmatrix], tuple[float, float]]:
    """
    Rescale a Hamiltonian to the interval ``[-1 - eps/2, 1 + eps/2]``.

    Adapted with modifications from kwant.kpm
    Copyright 2011-2016 Kwant developers, BSD simplified license
    https://gitlab.kwant-project.org/kwant/kwant/-/blob/v1.4.3/LICENSE.rst

    Parameters
    ----------
    hamiltonian :
        Hamiltonian of the system.
    eps :
        Extra tolerance to add to the spectral bounds as a fraction of the bandwidth.
    bounds :
        Estimated boundaries of the spectrum. If not provided the maximum and
        minimum eigenvalues are calculated.

    Returns
    -------
    hamiltonian_rescaled :
        Rescaled Hamiltonian.
    (a, b) :
        Rescaling parameters such that ``h_rescaled = (h - b) / a``.
    """

    if bounds is not None:
        lmin, lmax = bounds
    else:
        # Relative tolerance to which to calculate eigenvalues.  Because after
        # rescaling we will add eps / 2 to the spectral bounds, we don't need
        # to know the bounds more accurately than eps / 2.
        tol = eps / 2
        lmax, lmin = [
            sparse.linalg.eigsh(
                hamiltonian, k=1, which=which, return_eigenvectors=False, tol=tol
            )[0]
            for which in ("LA", "SA")
        ]

        if lmax - lmin <= abs(lmax + lmin) * tol / 2:
            raise ValueError(
                "The Hamiltonian has a single eigenvalue, it is not possible to "
                "obtain a spectral density."
            )

    a = np.abs(lmax - lmin) / (2.0 - eps)
    b = (lmax + lmin) / 2.0

    if sparse.issparse(hamiltonian):
        identity = sparse.identity(hamiltonian.shape[0], format="csr")
        rescaled_h = (hamiltonian - b * identity) / a
    elif isinstance(hamiltonian, np.ndarray):
        rescaled_h = (hamiltonian - b * np.eye(hamiltonian.shape[0])) / a
    else:
        raise TypeError("hamiltonian must be a numpy array or a sparse matrix")
    return rescaled_h, (a, b)


def jackson_kernel(N: int) -> np.ndarray:
    """
    Coefficients of the Jackson kernel of length N.

    Taken from Eq. (71) of `Rev. Mod. Phys., Vol. 78, No. 1 (2006)
    <https://arxiv.org/abs/cond-mat/0504627>`_.
    """
    m = np.arange(N) / (N + 1)
    denom = (N + 1) * np.tan(np.pi / (N + 1))
    return (1 - m) * np.cos(m * np.pi) + np.sin(m * np.pi) / denom
