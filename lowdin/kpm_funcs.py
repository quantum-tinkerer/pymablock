from typing import Optional

from scipy import sparse
import numpy as np
from kwant.kpm import _rescale, jackson_kernel


def greens_function(
    ham: np.ndarray | sparse.spmatrix,
    vectors: np.ndarray,
    params: Optional[dict] = None,
    kpm_params: Optional[dict] = None,
):
    """Build a Green's function operator using KPM.

    Returns a function that takes an energy or a list of energies, and returns
    the Green's function with that energy acting on `vectors`.

    Parameters
    ----------
    ham :
        Hamiltonian with shape `(N, N)`.
    vectors :
        Vectors upon which the Green's function will act.
        Vector of length `N` or array of vectors with shape `(M, N)`.
        `M` is the number of vectors and `N` the number of orbitals
        in the system.
    params :
        Parameters for the kwant system if ``ham`` is a kwant system.
    kpm_params :
        Dictionary containing the parameters to pass to the `~kwant.kpm`
        module. 'num_vectors' will be overwritten to match the number
        of vectors, and 'operator' key will be deleted.

    Returns
    -------
    green_expansion : callable
        Takes an energy or array of energies and returns the Greens function
        acting on the vectors, for those energies.
        The ndarray returned has initial dimension the same `num_e` as `e`,
        unless `e` is a scalar, and second dimension `M` unless `vectors` is
        a single vector. The shape of the returned array is `(num_e, M, N)` or
        `(M, N)` or `(num_e, N)` or `(N,)`.
    """
    if kpm_params is None:
        kpm_params = {}
    vectors = np.atleast_2d(vectors)

    # Rescale Hamiltonian
    ham, (_a, _b), num_moments, kernel = _kpm_preprocess(ham, kpm_params)

    # Get the kernel coefficients
    m = np.arange(num_moments)
    gs = kernel(np.ones(num_moments))
    gs[0] = gs[0] / 2

    def green_expansion(e):
        """
        Takes an energy ``e`` and returns the Green's function evaluated at
        that energy times the vectors.

        Parameters
        ----------
        e : float or array-like
            Energy or array of energies at which to evaluate the Green's
            function.

        Returns
        -------
        vecs_in_energy : ndarray
            Initial dimensions '(N_E, M, N)', where 'N_E' is the number of
            energies passed, 'M' is the number of vectors, and 'N' is the
            dimension of the vectors.
        """
        # remember if only one e was given
        e = np.atleast_1d(e).flatten()
        e_rescaled = (e - _b) / _a
        phi_e = np.arccos(e_rescaled)
        prefactor = -2j / (np.sqrt((1 - e_rescaled) * (1 + e_rescaled)))
        prefactor = prefactor / _a  # rescale energy expansion
        coef = gs[:, None] * np.exp(-1j * np.outer(m, phi_e))
        coef = prefactor * coef

        # Make generator to calculate expanded vectors on the fly
        expanded_vector_generator = _kpm_vectors(ham, vectors, num_moments)
        # Order as (energies, vectors, degrees of freedom)
        vecs_in_energy = sum(
            vec[None, :, :].T * c[None, None, :]
            for c, vec in zip(coef, expanded_vector_generator)
        )

        return vecs_in_energy

    return green_expansion


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
