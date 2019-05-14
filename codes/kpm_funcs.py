from functools import partial
from numbers import Number
import scipy
import numpy as np

import kwant
from kwant._common import ensure_rng


def build_greens_function(ham, vectors, params=None, kpm_params=dict(),
                          precalculate_moments=False):
    """Build a Green's function operator using KPM.

    Returns a function that takes an energy or a list of energies, and returns
    the Green's function with that energy acting on `vectors`.

    Parameters
    ----------
    ham : kwant.System or ndarray
        Finalized kwant system or dense or sparse ndarray of the
        Hamiltonian with shape `(N, N)`.
    vectors : 1D or 2D array
        Vectors upon which the Green's function will act.
        Vector of length `N` or array of vectors with shape `(M, N)`.
        `M` is the number of vectors and `N` the number of orbitals
        in the system.
    params : dict, optional
        Parameters for the kwant system.
    kpm_params : dict, optional
        Dictionary containing the parameters to pass to the `~kwant.kpm`
        module. 'num_vectors' will be overwritten to match the number
        of vectors, and 'operator' key will be deleted.
    precalculate_moments: bool, default False
        Whether to precalculate and store all the KPM moments of `vectors`.
        This is useful if the Green's function is evaluated at a large
        number of energies, but uses a large amount of memory.
        If False, the KPM expansion is performed every time the Green's
        function is called, which minimizes memory use.

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
    # remember if only one vector is given
    if len(vectors.shape) == 1:
        one_vec = True
        vectors = np.atleast_2d(vectors)
    elif len(vectors.shape) > 2 or len(vectors.shape) == 0:
        raise ValueError('vectors must be a 1D or 2D array.')
    else:
        one_vec = False
    num_vectors, dim = vectors.shape
    # prefactors of the kernel in kpm
    kernel = kpm_params.get('kernel', kwant.kpm.jackson_kernel)
    # Normalize the format of `ham`
    if isinstance(ham, kwant.system.System):
        ham = ham.hamiltonian_submatrix(params=params, sparse=True)
    try:
        ham = scipy.sparse.csr_matrix(ham)
    except Exception:
        raise ValueError("'ham' is neither a matrix nor a Kwant system.")

    # precalclulate expanded_vectors
    if precalculate_moments:
        kpm_params['num_vectors'] = num_vectors
        kpm_params['vector_factory'] = vectors
        # overwrite operator to extract kpm expanded vectors only
        kpm_params['operator'] = lambda bra, ket: ket

        # calculate kpm expanded vectors
        spectrum = kwant.kpm.SpectralDensity(ham, **kpm_params)
        _a, _b = spectrum._a, spectrum._b
        expanded_vectors = np.array(spectrum._moments_list)
        expanded_vectors = np.moveaxis(expanded_vectors, -1, 0)
        num_moments = spectrum.num_moments
    else:
        # Find the bounds of the spectrum and rescale `ham`
        eps = kpm_params.get('eps', 0.05)
        bounds = kpm_params.get('bounds', None)
        if eps <= 0:
            raise ValueError("'eps' must be positive")
        # Hamiltonian rescaled as in Eq. (24)
        ham, (_a, _b) = kwant.kpm._rescale(ham, eps=eps, bounds=bounds, v0=None)
        # extract the number of moments or set default to 100
        energy_resolution = kpm_params.get('energy_resolution')
        if energy_resolution is not None:
            num_moments = int(np.ceil((1.6 * _a) / energy_resolution))
            if kpm_params.get('num_moments'):
                raise TypeError("Only one of 'num_moments' or 'energy_resolution' can be provided.")
        elif kpm_params.get('num_moments') is None:
            num_moments = 100
        else:
            num_moments = kpm_params.get('num_moments')

    # Get the kernel coefficients
    m = np.arange(num_moments)
    gs = kernel(np.ones(num_moments))
    gs[0] = gs[0] / 2

    def green_expansion(e):
        """Takes an energy and returns the Greens function times the vectors,
        for those energies.

        The ndarray returned has initial dimension the same `num_e` as `e`,
        unless `e` is a scalar, and second dimension `M` unless `vectors` is
        a single vector. The shape of the returned array is `(num_e, M, N)` or
        `(M, N)` or `(num_e, N)` or `(N,)`.
        """
        # remember if only one e was given
        one_e = isinstance(e, Number)
        e = np.atleast_1d(e).flatten()
        e_rescaled = (e - _b) / _a
        phi_e = np.arccos(e_rescaled)
        prefactor = -2j / (np.sqrt(1 - e_rescaled**2))
        prefactor = prefactor / _a # rescale energy expansion
        coef = gs[:, None] * np.exp(-1j * np.outer(m, phi_e))
        coef = prefactor * coef

        if precalculate_moments:
            expanded_vectors_in_energy = expanded_vectors @ coef
        else:
            # Make generator to calculate expanded vectors on the fly
            expanded_vector_generator = _kpm_vector_generator(ham, vectors, num_moments)
            # Make sure axes in the result are ordered as (energies, vectors, degrees of freedom)
            expanded_vectors_in_energy = sum(vec[None, :, :].T * c[None, None, :]
                                             for c, vec in zip(coef, expanded_vector_generator))

        return expanded_vectors_in_energy

    return green_expansion


def _kpm_vector_generator(ham, vectors, max_moments):
    """
    Generator object that succesively yields KPM vectors `T_n(ham) |vector>`
    for vectors in `vectors` for n in [0, max_moments].

    Parameters
    ----------
    vectors : 1D or 2D array
        Vector of length `N` or array of vectors with shape `(M, N)`. The size of
        the last index should be the same as the size of `ham` `N`.
    ham : 2D array
        Hamiltonian, dense or sparse array with shape (N, N)
    max_moments : int
        Number of moments to stop with iteration

    Notes
    -----
    Returns a sequence of expanded vectors of shape (M, N). If the input was
    a vector, M=1.
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
