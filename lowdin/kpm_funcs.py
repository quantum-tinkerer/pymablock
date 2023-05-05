from typing import Optional

from scipy import sparse
from scipy.sparse.linalg import LinearOperator
import numpy as np


def greens_function(
    ham: np.ndarray | sparse.spmatrix,
    energy: float,
    vector: np.ndarray,
    num_moments: int = 100,
    energy_resolution: Optional[float] = None,
):
    """Return a solution of the linear system (ham - energy) * x = vector.

    Uses the Kernel polynomial method (KPM) with the Jackson kernel.

    Parameters
    ----------
    ham :
        Hamiltonian with shape `(N, N)`. It is required that the Hamiltonian
        is rescaled so that its spectrum lies in the interval `[-1, 1]`.
    energy :
        Rescaled energy at which to evaluate the Green's function.
    vector :
        Vector with shape `(N,)`.
    num_moments :
        Number of moments to use in the KPM expansion.
    energy_resolution :
        Energy resolution to use in the KPM expansion. If specified, the
        number of moments is chosen automatically.

    Returns
    -------
    solution : `~np.ndarray`
        Solution of the linear system.
    """
    if energy_resolution is not None:
        num_moments = int(np.ceil(1.6 / energy_resolution))

    gs = jackson_kernel(num_moments)
    gs[0] = gs[0] / 2

    phi_e = np.arccos(energy)
    prefactor = -2j / (np.sqrt((1 - energy) * (1 + energy)))
    coef = gs * np.exp(-1j * np.arange(num_moments) * phi_e)
    coef = prefactor * coef

    return sum(vec * c for c, vec in zip(coef, kpm_vectors(ham, vector)))


def kpm_vectors(
    ham: np.ndarray,
    vectors: np.ndarray,
):
    r"""
    Generator of KPM vectors

    This generator yields vectors as
    .. math::
      T_n(H) \rvert v\langle

    for vectors :math:`\rvert v\langle` in ``vectors``,
    for 'n' in '[0, max_moments]'.

    Parameters
    ----------
    ham :
        Hamiltonian, dense or sparse array with shape '(N, N)'.
    vectors :
        Vector of length 'N' or array of vectors with shape '(M, N)'.

    Yields
    ------
    expanded_vectors : Iterable
        Sequence of expanded vectors of shape '(M, N)'.
        If the input is a vector then 'M=1'.
    """
    # Internally store as column vectors
    vectors = np.atleast_2d(vectors).T
    alpha_prev = np.zeros(vectors.shape, dtype=complex)
    alpha = np.zeros(vectors.shape, dtype=complex)
    alpha_next = np.zeros(vectors.shape, dtype=complex)

    alpha[:] = vectors
    yield alpha.T
    alpha_prev[:] = alpha
    alpha[:] = ham @ alpha
    yield alpha.T
    while True:
        alpha_next[:] = 2 * ham @ alpha - alpha_prev
        alpha_prev[:] = alpha
        alpha[:] = alpha_next
        yield alpha.T


def rescale(
    hamiltonian: np.ndarray | sparse.spmatrix | LinearOperator,
    eps: Optional[float] = 0.01,
    bounds: Optional[tuple[float, float]] = None,
):
    """Rescale a Hamiltonian to the interval ``[-1 - eps/2, 1 + eps/2]``.

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
        lmax = sparse.linalg.eigsh(
            hamiltonian, k=1, which="LA", return_eigenvectors=False, tol=tol
        )[0]
        lmin = sparse.linalg.eigsh(
            hamiltonian, k=1, which="SA", return_eigenvectors=False, tol=tol
        )[0]

        if lmax - lmin <= abs(lmax + lmin) * tol / 2:
            raise ValueError(
                "The Hamiltonian has a single eigenvalue, it is not possible to "
                "obtain a spectral density."
            )

    a = np.abs(lmax - lmin) / (2.0 - eps)
    b = (lmax + lmin) / 2.0

    if sparse.issparse(hamiltonian):
        identity = sparse.identity(hamiltonian.shape[0], format="csr")
        rescaled_ham = (hamiltonian - b * identity) / a
    elif isinstance(hamiltonian, np.ndarray):
        rescaled_ham = (hamiltonian - b * np.eye(hamiltonian.shape[0])) / a
    else:
        rescaled_ham = LinearOperator(
            shape=hamiltonian.shape, matvec=(lambda v: (hamiltonian @ v - b * v) / a)
        )

    return rescaled_ham, (a, b)


def jackson_kernel(N):
    """Coefficients of the Jackson kernel of length N.

    Taken from Eq. (71) of `Rev. Mod. Phys., Vol. 78, No. 1 (2006)
    <https://arxiv.org/abs/cond-mat/0504627>`_.
    """
    m = np.arange(N) / (N + 1)
    denom = (N + 1) * np.tan(np.pi / (N + 1))
    return (1 - m) * np.cos(m * np.pi) + np.sin(m * np.pi) / denom
