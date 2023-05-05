from typing import Optional

from scipy import sparse
import numpy as np
from kwant.kpm import _rescale, jackson_kernel


def greens_function(
    ham: np.ndarray | sparse.spmatrix,
    energy: float,
    vector: np.ndarray,
    kpm_params: Optional[dict] = None,
):
    """Return a solution of the linear system (ham - energy) * x = vector.

    Uses the Kernel polynomial method (KPM) with the Jackson kernel.

    Parameters
    ----------
    ham :
        Hamiltonian with shape `(N, N)`.
    energy :
        Energy at which to evaluate the Green's function.
    vector :
        Vector with shape `(N,)`.
    kpm_params :
        Dictionary containing the parameters to pass to the `~kwant.kpm`
        module. ``num_vectors`` and ``operator`` parameters are overridden.

    Returns
    -------
    solution : `~np.ndarray`
        Solution of the linear system.
    """
    if kpm_params is None:
        kpm_params = {}

    # Rescale Hamiltonian
    ham, (_a, _b), num_moments, kernel = _kpm_preprocess(ham, kpm_params)

    # Get the kernel coefficients
    m = np.arange(num_moments)
    gs = kernel(np.ones(num_moments))
    gs[0] = gs[0] / 2

    e_rescaled = (energy - _b) / _a
    phi_e = np.arccos(e_rescaled)
    prefactor = -2j / (np.sqrt((1 - e_rescaled) * (1 + e_rescaled)))
    prefactor = prefactor / _a  # rescale energy expansion
    coef = gs[:, None] * np.exp(-1j * np.outer(m, phi_e))
    coef = prefactor * coef

    return sum(vec * c for c, vec in zip(coef, _kpm_vectors(ham, vector, num_moments)))


def _kpm_vectors(
    ham: np.ndarray,
    vectors: np.ndarray,
    max_moments: int,
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
    max_moments :
        Number of moments to stop the iteration.

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
    n = 0
    yield alpha.T
    n += 1
    alpha_prev[:] = alpha
    alpha[:] = ham @ alpha
    yield alpha.T
    n += 1
    while n < max_moments:
        alpha_next[:] = 2 * ham @ alpha - alpha_prev
        alpha_prev[:] = alpha
        alpha[:] = alpha_next
        yield alpha.T
        n += 1


def _kpm_preprocess(ham, kpm_params):
    # Find the bounds of the spectrum and rescale `ham`
    eps = kpm_params.get("eps", 0.05)
    bounds = kpm_params.get("bounds", None)
    if eps <= 0:
        raise ValueError("'eps' must be positive")
    # Hamiltonian rescaled as in Eq. (24), This returns a LinearOperator
    ham_rescaled, (_a, _b) = _rescale(ham, eps=eps, bounds=bounds, v0=None)
    # Make sure to return the same format
    if isinstance(ham, np.ndarray):
        ham_rescaled = (ham - _b * np.eye(ham.shape[0])) / _a
    elif sparse.issparse(ham):
        id = sparse.identity(ham.shape[0], dtype="complex", format="csr")
        id = sparse.csr_array(id)
        ham_rescaled = (ham - _b * id) / _a
    # extract the number of moments or set default to 100
    energy_resolution = kpm_params.get("energy_resolution")
    if energy_resolution is not None:
        num_moments = int(np.ceil((1.6 * _a) / energy_resolution))
        if kpm_params.get("num_moments"):
            raise TypeError(
                "Either 'num_moments' or 'energy_resolution' can be provided."
            )
    elif kpm_params.get("num_moments") is None:
        num_moments = 100
    else:
        num_moments = kpm_params.get("num_moments")
    kernel = kpm_params.get("kernel", jackson_kernel)

    return ham_rescaled, (_a, _b), num_moments, kernel
